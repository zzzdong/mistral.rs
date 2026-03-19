#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::SdpaParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMasker, MatMul, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};

// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

//=============================================================================
// MLP - 与非量化版本保持一致
//=============================================================================
struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>, // gate_proj
    feed_forward_w2: Arc<dyn QuantMethod>, // down_proj
    feed_forward_w3: Arc<dyn QuantMethod>, // up_proj
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 修复：确保所有操作都在 xs 所在的设备上
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;

        // 确保 w1 和 w3 在同一设备上
        let w1 = w1.to_device(w3.device())?;

        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        MatMul.qmethod_matmul(&y, &*self.feed_forward_w2)
    }
}

//=============================================================================
// Full Attention Layer
//=============================================================================
struct FullAttentionWeights {
    attention_wq: Arc<dyn QuantMethod>, // projects to n_head * head_dim * 2 (Q + gate)
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rot_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl FullAttentionWeights {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        // Q projection outputs [n_head * head_dim * 2]
        let q_gate = MatMul.qmethod_matmul(x, &*self.attention_wq)?;

        // Split into Q and gate
        let q_gate = q_gate.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        // Gate: [batch, seq, n_head, head_dim] -> [batch, seq, n_head * head_dim]
        let gate = gate.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

        // K and V projections (normal)
        let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?;
        let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?;

        // Reshape K and V
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Apply Q/K normalization
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        // Apply RoPE (partial)
        let (q, k) = if self.rot_dim < self.head_dim {
            let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
            let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;

            let (q_rot, k_rot) = self.rotary.forward(&q_rot, &k_rot, start_offsets)?;
            let q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
            let k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
            (q, k)
        } else {
            self.rotary.forward(&q, &k, start_offsets)?
        };

        // 关键修复：在存入 kv_cache 之前，确保数据类型与 cache 一致
        let cache_dtype = match kv_cache {
            KvCache::Normal { k: kc, .. } => {
                if let Some(ref cache_tensor) = kc.all_data {
                    cache_tensor.dtype()
                } else {
                    self.dtype
                }
            }
            _ => self.dtype,
        };

        // 将 q/k/v 转换为 cache 的数据类型
        let q = q.to_dtype(cache_dtype)?;
        let k = k.to_dtype(cache_dtype)?;
        let v = v.to_dtype(cache_dtype)?;

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        // Reshape output
        let y = if mask.is_some() {
            y.transpose(1, 2)?
                .reshape((b_sz, seq_len, self.n_head * self.head_dim))?
        } else {
            y.reshape((b_sz, seq_len, self.n_head * self.head_dim))?
        };

        // Apply gate with sigmoid - gate 是 F32，需要转换
        let gate_sigmoid =
            candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(y.dtype())?;
        let y = y.broadcast_mul(&gate_sigmoid)?;

        // Output projection
        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)
    }
}

//=============================================================================
// Gated Delta Net (Linear Attention)
//=============================================================================
struct GatedDeltaNetWeights {
    // Projections
    in_proj_qkv: Arc<dyn QuantMethod>, // projects to key_dim*2 + value_dim
    in_proj_z: Arc<dyn QuantMethod>,   // projects to value_dim (gate)
    in_proj_beta: Arc<dyn QuantMethod>, // projects to num_v_heads
    in_proj_alpha: Arc<dyn QuantMethod>, // projects to num_v_heads
    out_proj: Arc<dyn QuantMethod>,    // projects back to hidden_size

    // Parameters (dequantized)
    conv1d_weight: Tensor, // [conv_dim, 1, kernel_size]
    dt_bias: Tensor,       // [num_v_heads]
    a_log: Tensor,         // [num_v_heads] (log of A)
    norm: QRmsNorm,        // Gated RMSNorm (weight only)

    // Dimensions
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    inner_size: usize, // 新增：linear_inner_size
    dt_rank: usize,    // 新增：linear_dt_rank

    // States (for caching)
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNetWeights {
    fn new<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        inner_size: usize, // 新增参数
        dt_rank: usize,    // 新增参数
        rms_norm_eps: f32,
    ) -> Result<Self> {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        // 验证维度是否匹配
        if value_dim != inner_size {
            candle_core::bail!(
                "Value dimension mismatch: value_dim={}, inner_size={}",
                value_dim,
                inner_size
            );
        }

        // dt_rank 应该等于 num_v_heads
        if dt_rank != num_v_heads {
            candle_core::bail!(
                "dt_rank mismatch: dt_rank={}, num_v_heads={}",
                dt_rank,
                num_v_heads
            );
        }

        // 加载量化投影
        let in_proj_qkv = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
        let in_proj_z = ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?;
        let in_proj_beta = ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?;
        let in_proj_alpha = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?;
        let out_proj = ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?;

        // 加载并反量化参数
        let conv1d_weight = ct
            .tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
            .dequantize(device)?
            .reshape((conv_dim, 1, conv_kernel_size))?;

        let a_log = ct
            .tensor(&format!("{prefix}.ssm_a"), device)?
            .dequantize(device)?;

        let dt_bias = ct
            .tensor(&format!("{prefix}.ssm_dt.bias"), device)?
            .dequantize(device)?;

        let norm_weight = ct.tensor(&format!("{prefix}.ssm_norm.weight"), device)?;

        // 验证 dt_bias 的形状
        if dt_bias.dim(0)? != num_v_heads {
            candle_core::bail!(
                "dt_bias shape mismatch: expected {}, got {}",
                num_v_heads,
                dt_bias.dim(0)?
            );
        }

        // 验证 a_log 的形状
        if a_log.dim(0)? != num_v_heads {
            candle_core::bail!(
                "a_log shape mismatch: expected {}, got {}",
                num_v_heads,
                a_log.dim(0)?
            );
        }

        Ok(Self {
            in_proj_qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_qkv),
                b: None,
            })?),
            in_proj_z: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_z),
                b: None,
            })?),
            in_proj_beta: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_beta),
                b: None,
            })?),
            in_proj_alpha: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_alpha),
                b: None,
            })?),
            out_proj: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(out_proj),
                b: None,
            })?),
            conv1d_weight,
            dt_bias,
            a_log,
            norm: QRmsNorm::new(norm_weight, rms_norm_eps)?,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            inner_size,
            dt_rank,
            conv_state: None,
            recurrent_state: None,
        })
    }

    // L2 normalization
    fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let norm = (x_f32.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
        let result = x_f32.broadcast_div(&norm)?;
        result.to_dtype(x.dtype())
    }

    // Core delta net recurrence
    fn delta_net_recurrence(
        &self,
        q: &Tensor,    // [batch, seq, num_v_heads, head_k_dim]
        k: &Tensor,    // [batch, seq, num_v_heads, head_k_dim]
        v: &Tensor,    // [batch, seq, num_v_heads, head_v_dim]
        gate: &Tensor, // [batch, seq, num_v_heads]
        beta: &Tensor, // [batch, seq, num_v_heads]
        state: &mut Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, num_heads, head_k_dim) = q.dims4()?;
        let head_v_dim = v.dim(3)?;

        // 验证维度一致性
        if num_heads != self.num_v_heads {
            candle_core::bail!(
                "Head count mismatch in recurrence: expected {}, got {}",
                self.num_v_heads,
                num_heads
            );
        }

        // Convert to f32 for stable computation
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;
        let v_f32 = v.to_dtype(DType::F32)?;
        let gate_f32 = gate.to_dtype(DType::F32)?;
        let beta_f32 = beta.to_dtype(DType::F32)?;

        let scale = 1.0 / (head_k_dim as f64).sqrt();
        let q_f32 = (q_f32 * scale)?;

        let mut outputs = Vec::with_capacity(seq_len);
        let mut current_state = state.clone();

        for t in 0..seq_len {
            let q_t = q_f32.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_k_dim]
            let k_t = k_f32.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_k_dim]
            let v_t = v_f32.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads, head_v_dim]
            let g_t = gate_f32.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads]
            let b_t = beta_f32.narrow(1, t, 1)?.squeeze(1)?; // [batch, num_heads]

            // 1. Apply decay: state = state * exp(g_t)
            let g_t_exp = g_t.unsqueeze(2)?.unsqueeze(3)?.exp()?; // [batch, num_heads, 1, 1]
            current_state = current_state.broadcast_mul(&g_t_exp)?;

            // 2. kv_mem = k_t^T @ state
            let k_t_4d = k_t.unsqueeze(3)?; // [batch, num_heads, head_k_dim, 1]
            let kv_mem = k_t_4d.transpose(2, 3)?.matmul(&current_state)?; // [batch, num_heads, 1, head_v_dim]
            let kv_mem = kv_mem.squeeze(2)?; // [batch, num_heads, head_v_dim]

            // 3. delta = (v_t - kv_mem) * beta_t
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&b_t.unsqueeze(2)?)?; // [batch, num_heads, head_v_dim]

            // 4. state = state + k_t * delta
            let delta_4d = delta.unsqueeze(2)?; // [batch, num_heads, 1, head_v_dim]
            let state_update = k_t_4d.matmul(&delta_4d)?; // [batch, num_heads, head_k_dim, head_v_dim]
            current_state = (current_state + state_update)?;

            // 5. out_t = q_t @ state
            let q_t_4d = q_t.unsqueeze(2)?; // [batch, num_heads, 1, head_k_dim]
            let out_t = q_t_4d.matmul(&current_state)?.squeeze(2)?; // [batch, num_heads, head_v_dim]
            outputs.push(out_t);
        }

        *state = current_state;

        // Stack outputs: [batch, seq, num_heads, head_v_dim]
        let output = Tensor::stack(&outputs, 1)?;
        output.to_dtype(q.dtype())
    }

    // Conv1d for prefill (multiple tokens)
    fn conv1d_prefill(&mut self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        // x: [batch, conv_dim, seq_len]
        let pad = self.conv_kernel_size - 1;
        let device = x.device();
        let dtype = x.dtype();

        // Left padding
        let padding = Tensor::zeros(
            (x.dim(0)?, self.key_dim * 2 + self.value_dim, pad),
            dtype,
            device,
        )?;
        let padded = Tensor::cat(&[padding, x.clone()], 2)?;

        // Save state for future steps - 确保与 x 的数据类型一致
        let state_slice = padded.narrow(2, seq_len, pad)?;

        // 强制转换为 x 的数据类型
        let state_slice = state_slice.to_dtype(dtype)?;

        self.conv_state = Some(state_slice);

        // Apply conv1d
        let conv_out = padded.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv1d_weight.dim(0)?)?;
        candle_nn::ops::silu(&conv_out)
    }

    // Conv1d for single token generation
    fn conv1d_step(&mut self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, conv_dim, 1]
        let dtype = x.dtype();
        let device = x.device();

        let conv_state = self.conv_state.as_ref().unwrap();

        // 确保 conv_state 与 x 的数据类型一致
        let conv_state = if conv_state.dtype() != dtype {
            conv_state.to_dtype(dtype)?.to_device(device)?
        } else {
            conv_state.to_device(device)?
        };

        let combined = Tensor::cat(&[&conv_state, x], 2)?;

        // Update state (keep last kernel_size-1 elements) - 确保与 x 的数据类型一致
        let new_state = combined.narrow(2, 1, self.conv_kernel_size - 1)?;

        // 强制转换为 x 的数据类型
        let new_state = new_state.to_dtype(dtype)?;

        self.conv_state = Some(new_state);

        // Apply conv1d
        let conv_out = combined.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv1d_weight.dim(0)?)?;
        candle_nn::ops::silu(&conv_out)
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device();

        // 验证 hidden_size 匹配 inner_size
        if hidden_size != self.inner_size {
            candle_core::bail!(
                "Hidden size mismatch: expected {}, got {}",
                self.inner_size,
                hidden_size
            );
        }

        // 1. Input projections - 确保输出保持原始数据类型
        let qkvz = MatMul.qmethod_matmul(x, &*self.in_proj_qkv)?;
        let z = MatMul.qmethod_matmul(x, &*self.in_proj_z)?;
        let beta_raw = MatMul.qmethod_matmul(x, &*self.in_proj_beta)?;
        let alpha_raw = MatMul.qmethod_matmul(x, &*self.in_proj_alpha)?;

        // 2. Split QKVZ
        let value_dim = self.num_v_heads * self.head_v_dim;
        let q = qkvz.narrow(D::Minus1, 0, self.key_dim)?;
        let k = qkvz.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = qkvz.narrow(D::Minus1, self.key_dim * 2, value_dim)?;

        // 3. Prepare for conv1d
        let conv_input = Tensor::cat(&[q, k, v], D::Minus1)?; // [batch, seq, conv_dim]
        let conv_input_t = conv_input.transpose(1, 2)?; // [batch, conv_dim, seq]

        let conv_out = if seq_len == 1 && self.conv_state.is_some() {
            self.conv1d_step(&conv_input_t)?
        } else {
            self.conv1d_prefill(&conv_input_t, seq_len)?
        };

        let conv_out = conv_out.transpose(1, 2)?; // [batch, seq, conv_dim]

        // 4. Split convolved outputs
        let q_conv = conv_out.narrow(2, 0, self.key_dim)?;
        let k_conv = conv_out.narrow(2, self.key_dim, self.key_dim)?;
        let v_conv = conv_out.narrow(2, self.key_dim * 2, value_dim)?;

        // 5. Reshape to separate heads
        let mut q = q_conv.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let mut k = k_conv.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 6. Compute beta and gate
        let beta = candle_nn::ops::sigmoid(&beta_raw)?;
        let beta = beta.reshape((batch_size, seq_len, self.num_v_heads))?;

        // gate = -exp(A_log) * softplus(alpha + dt_bias)
        let a_log_f32 = self.a_log.to_dtype(DType::F32)?;
        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let alpha_f32 = alpha_raw.to_dtype(DType::F32)?;

        let alpha_biased = alpha_f32.broadcast_add(&dt_bias_f32.unsqueeze(0)?.unsqueeze(0)?)?;
        let softplus = (alpha_biased.exp()? + 1.0)?.log()?;
        let neg_a_exp = a_log_f32.exp()?.neg()?;
        let gate = neg_a_exp.broadcast_mul(&softplus)?.to_dtype(dtype)?;
        let gate = gate.reshape((batch_size, seq_len, self.num_v_heads))?;

        // 7. Repeat K/Q if num_v_heads != num_k_heads
        if self.num_v_heads != self.num_k_heads {
            let repeat_factor = self.num_v_heads / self.num_k_heads;

            q = q
                .unsqueeze(3)?
                .repeat((1, 1, 1, repeat_factor, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;

            k = k
                .unsqueeze(3)?
                .repeat((1, 1, 1, repeat_factor, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
        }

        // 8. L2 normalize Q and K - 确保输出数据类型一致
        let q = Self::l2_norm(&q, 1e-6)?.to_dtype(dtype)?;
        let k = Self::l2_norm(&k, 1e-6)?.to_dtype(dtype)?;

        // 9. Delta net recurrence
        let mut state = if let Some(state) = &self.recurrent_state {
            state.to_device(device)?
        } else {
            Tensor::zeros(
                (
                    batch_size,
                    self.num_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                ),
                DType::F32,
                device,
            )?
        };

        let attn_out = self.delta_net_recurrence(&q, &k, &v, &gate, &beta, &mut state)?;

        // 保存 state 时保持 F32
        self.recurrent_state = Some(state);

        // 10. Gated RMSNorm
        let attn_out_flat =
            attn_out.reshape((batch_size * seq_len, self.num_v_heads, self.head_v_dim))?;
        let z_flat = z.reshape((batch_size * seq_len, self.num_v_heads, self.head_v_dim))?;

        // Apply norm and gate
        let mut normed = Vec::with_capacity(batch_size * seq_len);
        for i in 0..batch_size * seq_len {
            let attn_i = attn_out_flat.narrow(0, i, 1)?.squeeze(0)?;
            let z_i = z_flat.narrow(0, i, 1)?.squeeze(0)?;

            // RMSNorm on attn_out
            let normed_i = self.norm.forward(&attn_i)?;

            // Apply gate with SiLU - 确保 gate 的数据类型正确
            let z_i_f32 = z_i.to_dtype(DType::F32)?;
            let z_silu = candle_nn::ops::silu(&z_i_f32)?.to_dtype(normed_i.dtype())?;
            let gated = normed_i.broadcast_mul(&z_silu)?;
            normed.push(gated);
        }

        let normed = Tensor::stack(&normed, 0)?;
        let normed = normed.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;

        // 11. Output projection - 确保输出数据类型与输入一致
        let normed = normed.to_dtype(dtype)?;

        MatMul.qmethod_matmul(&normed, &*self.out_proj)
    }
}

//=============================================================================
// Layer Weights - 统一包含两种注意力类型
//=============================================================================
enum AttentionType {
    Full(FullAttentionWeights),
    Linear(GatedDeltaNetWeights),
}

struct LayerWeights {
    attn_type: AttentionType,
    mlp: Mlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

impl LayerWeights {
    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let attn = match &mut self.attn_type {
            AttentionType::Full(attn) => {
                attn.forward(&x, mask, start_offsets, kv_cache, metadata)?
            }
            AttentionType::Linear(attn) => attn.forward(&x)?,
        };

        let x = (attn + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

//=============================================================================
// Model Configuration Properties
//=============================================================================
pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub full_attention_interval: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_inner_size: usize,
    pub linear_dt_rank: usize,
}

fn verify_qwen35_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen3" && actual_arch != "qwen35" {
        candle_core::bail!("Expected `qwen3` or `qwen35` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let arch = verify_qwen35_arch(c.metadata)?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // Qwen3.5 specific metadata
        let full_attention_interval =
            c.get_value::<u32>("full_attention_interval").unwrap_or(4) as usize;

        let linear_num_key_heads = c.get_value::<u32>("ssm.group_count").unwrap_or(16) as usize;

        let linear_key_head_dim = c.get_value::<u32>("ssm.state_size").unwrap_or(128) as usize;

        let linear_inner_size = c.get_value::<u32>("ssm.inner_size").unwrap_or(2048) as usize;

        let linear_num_value_heads = linear_inner_size / linear_key_head_dim;

        let linear_value_head_dim =
            c.get_value::<u32>("ssm.value_head_dim").unwrap_or(128) as usize;

        let linear_conv_kernel_dim = c.get_value::<u32>("ssm.conv_kernel").unwrap_or(4) as usize;

        let linear_dt_rank = c
            .get_value::<u32>("ssm.dt_rank")
            .unwrap_or(linear_num_value_heads as u32) as usize;

        // 验证维度关系
        if linear_num_value_heads * linear_value_head_dim != linear_inner_size {
            anyhow::bail!(
                "Dimension mismatch: num_v_heads({}) * head_v_dim({}) = {} != inner_size({})",
                linear_num_value_heads,
                linear_value_head_dim,
                linear_num_value_heads * linear_value_head_dim,
                linear_inner_size
            );
        }

        // dt_rank 应该等于 num_v_heads
        if linear_dt_rank != linear_num_value_heads {
            anyhow::bail!(
                "dt_rank({}) should equal num_v_heads({})",
                linear_dt_rank,
                linear_num_value_heads
            );
        }

        // conv_kernel_dim 应该是有效的
        if linear_conv_kernel_dim < 2 || linear_conv_kernel_dim > 10 {
            anyhow::bail!(
                "conv_kernel_dim({}) out of expected range [2, 10]",
                linear_conv_kernel_dim
            );
        }

        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            full_attention_interval,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            linear_inner_size,
            linear_dt_rank,
        };

        Ok(props)
    }
}

//=============================================================================
// Main Model Weights
//=============================================================================
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen35_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };

        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
            full_attention_interval,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            linear_inner_size,
            linear_dt_rank,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        // Load embeddings
        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;

        // Load final norm and output
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        // Qwen3.5 uses partial_rotary_factor=0.25
        let partial_rotary_factor = 0.25_f64;
        let rot_dim = (head_dim as f64 * partial_rotary_factor) as usize;

        // Initialize rotary embeddings for full attention layers
        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            // Only full attention layers need RoPE (layers where (layer_idx+1) % interval == 0)
            if (layer_idx + 1) % full_attention_interval == 0 {
                ropes.insert(
                    device.location(),
                    Arc::new(RotaryEmbedding::new_partial(
                        rope_freq_base,
                        rot_dim,
                        max_seq_len,
                        device,
                        true,
                        DType::F32,
                    )?),
                );
            }
        }

        // Build layers
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let is_full_attention = (layer_idx + 1) % full_attention_interval == 0;

            let attn_type = if is_full_attention {
                let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

                let q_norm = QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let k_norm = QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                    rms_norm_eps,
                )?;

                let rotary = ropes
                    .get(&device.location())
                    .expect("No RoPE for device location!")
                    .clone();

                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(head_dim, device, None)?)
                    }
                };

                AttentionType::Full(FullAttentionWeights {
                    attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wq),
                        b: None,
                    })?),
                    attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wk),
                        b: None,
                    })?),
                    attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wv),
                        b: None,
                    })?),
                    attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wo),
                        b: None,
                    })?),
                    q_norm,
                    k_norm,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    rot_dim,
                    rotary,
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: head_count / head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                        sliding_window: None,
                        sinks: None,
                    },
                    dtype,
                })
            } else {
                AttentionType::Linear(GatedDeltaNetWeights::new(
                    &mut ct,
                    &prefix,
                    device,
                    linear_num_value_heads,
                    linear_num_key_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    linear_conv_kernel_dim,
                    linear_inner_size, // 传递 inner_size
                    linear_dt_rank,    // 传递 dt_rank
                    rms_norm_eps,
                )?)
            };

            // MLP
            let feed_forward_w1 = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_w2 = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
            let feed_forward_w3 = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
            let mlp = Mlp {
                feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w1),
                    b: None,
                })?),
                feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w2),
                    b: None,
                })?),
                feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(feed_forward_w3),
                    b: None,
                })?),
            };

            // Layer norms
            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?;

            layers.push(LayerWeights {
                attn_type,
                mlp,
                input_layernorm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                post_attention_layernorm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &mut self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let cache = &mut self.cache.normal().0;

        // Determine number of heads for mask creation
        let n_head = match &self.layers[0].attn_type {
            AttentionType::Full(ref attn) => attn.n_head,
            AttentionType::Linear(ref linear) => linear.num_v_heads,
        };

        let mask = CausalMasker.make_causal_mask_matrix(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            n_head,
        )?;

        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });

        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            // 修复：在进入每个层之前确保输入在正确的设备上
            if let Some(ref mapper) = self.mapper {
                let target_device = mapper.device_for(i, false).unwrap_or(&self.device);

                layer_in = layer_in.to_device(target_device)?;
            }

            layer_in = layer.forward(
                &layer_in,
                mask.as_ref().map(|m| m.get(layer_in.device())),
                start_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
            )?;
        }

        // 修复：确保在应用 final norm 之前将张量移动到正确的设备
        let final_device = self.norm.weight().device();
        let layer_in = layer_in.to_device(final_device)?;

        let x = self.norm.forward(&layer_in)?;

        // 修复：确保 logits 在 output 所在的设备上
        let x = extract_logits(&x, context_lens)?;
        let x_contiguous = x.contiguous()?;

        let x_final = x_contiguous;

        MatMul.qmethod_matmul(&x_final, &*self.output)
    }
}
