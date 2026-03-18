#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;
use mistralrs_quant::{
    ColumnParallelLayer, GgufMatMul, MatMul, QuantMethod, QuantMethodConfig, QuantizedConfig,
    RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};

use crate::models::gdn::{GdnConfig, GdnLayerCache};
use crate::serde_default_fn;
use crate::utils::model_config as ModelConfig;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    gguf::Content,
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
    },
    layers::{CausalMasker, QGemmaRmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalLoadingMetadata, NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};



const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

//=============================================================================
// Configuration
//=============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: crate::layers::Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub head_dim: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    // GDN config
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

serde_default_fn!(bool, default_tie, true);
serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);

#[derive(Debug, Clone)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

impl Config {
    pub fn layer_types(&self) -> Vec<LayerType> {
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

impl GdnConfig for Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
}

//=============================================================================
// Full Attention Layer (quantized)
//=============================================================================
struct FullAttention {
    wq: Arc<dyn QuantMethod>,
    wk: Arc<dyn QuantMethod>,
    wv: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    q_norm: QGemmaRmsNorm,
    k_norm: QGemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    rot_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    fn new<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Load quantized weights from GGUF
        let wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
        let wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
        let wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
        let wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

        // 加载 Q/K norms - 直接使用 QGemmaRmsNorm::new 从 QTensor 创建
        let q_norm_qt = ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?;
        let q_norm = QGemmaRmsNorm::new(q_norm_qt, cfg.rms_norm_eps as f32)?;

        let k_norm_qt = ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?;
        let k_norm = QGemmaRmsNorm::new(k_norm_qt, cfg.rms_norm_eps as f32)?;

        // Create quantized matmul wrappers
        let make_qmatmul = |qt: QTensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(qt),
                b: None,
            })?))
        };

        let rot_dim = (head_dim as f64 * cfg.partial_rotary_factor) as usize;

        let sliding_window = None;
        Ok(Self {
            wq: make_qmatmul(wq)?,
            wk: make_qmatmul(wk)?,
            wv: make_qmatmul(wv)?,
            wo: make_qmatmul(wo)?,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            rot_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let original_dtype = x.dtype();
        let mut x = x.clone();

        // Quantized activation type handling
        if let Some(t) = self.wq.quantized_act_type() {
            x = x.to_dtype(t)?;
        }

        let mut q_gate = MatMul.qmethod_matmul(&x, &*self.wq)?;
        let mut k = MatMul.qmethod_matmul(&x, &*self.wk)?;
        let mut v = MatMul.qmethod_matmul(&x, &*self.wv)?;

        if self.wq.quantized_act_type().is_some() {
            q_gate = q_gate.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        // Split q_gate into q and gate
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Reshape for attention
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Apply QK norm
        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        // Apply partial RoPE
        if self.rot_dim < self.head_dim {
            let q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
            let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;

            let (q_rot, k_rot) = self.rotary_emb.forward(&q_rot, &k_rot, seqlen_offsets)?;
            q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
            k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
        } else {
            let (q_new, k_new) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;
            q = q_new;
            k = k_new;
        }

        // Standard attention
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask.as_ref(),
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask.as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.wq.quantized_act_type() {
            y = y.to_dtype(t)?;
        }

        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let mut res = MatMul.qmethod_matmul(&y, &*self.wo)?;
        if self.wq.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

//=============================================================================
// Gated Delta Net from GGUF
//=============================================================================

#[derive(Debug, Clone)]
pub struct QGatedDeltaNet {
    // 维度参数
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    
    // 量化后的 projections
    in_proj_qkv: Arc<dyn QuantMethod>,  // QKV 合并投影
    in_proj_z: Arc<dyn QuantMethod>,     // gate 投影
    in_proj_beta: Arc<dyn QuantMethod>,  // beta 投影
    in_proj_alpha: Arc<dyn QuantMethod>, // alpha 投影
    out_proj: Arc<dyn QuantMethod>,      // 输出投影
    
    // 反量化后的参数（这些参数很小，可以直接存储为 Tensor）
    conv1d_weight: Tensor,  // [conv_dim, 1, kernel_size]
    dt_bias: Tensor,        // [num_v_heads]
    a_log: Tensor,          // [num_v_heads]
    norm: QGemmaRmsNormGated, // Gated RMSNorm
}

impl QGatedDeltaNet {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
        cfg: &Config,
    ) -> Result<Self> {
        let num_v_heads = cfg.linear_num_value_heads;
        let num_k_heads = cfg.linear_num_key_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;
        let conv_dim = key_dim * 2 + value_dim;

        // 1. 加载量化后的 projections
        let in_proj_qkv = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
        let in_proj_z = ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?;
        let in_proj_beta = ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?;
        let in_proj_alpha = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?;
        let out_proj = ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?;

        // 2. 加载并反量化 conv1d 权重（需要 reshape）
        let conv1d_weight = ct.tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
            .dequantize(device)?
            .reshape((conv_dim, 1, conv_kernel_size))?;

        // 3. 加载并反量化 SSM 参数
        let a_log = ct.tensor(&format!("{prefix}.ssm_a"), device)?.dequantize(device)?;
        let dt_bias = ct.tensor(&format!("{prefix}.ssm_dt.bias"), device)?.dequantize(device)?;

        // 4. 加载 gated norm
        let norm_weight = ct.tensor(&format!("{prefix}.ssm_norm.weight"), device)?;
        let norm = QGemmaRmsNormGated::new(norm_weight, cfg.rms_norm_eps as f32)?;

        // 5. 创建 quantized matmul wrappers
        let make_qmatmul = |qt: QTensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(qt),
                b: None,
            })?))
        };

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            in_proj_qkv: make_qmatmul(in_proj_qkv)?,
            in_proj_z: make_qmatmul(in_proj_z)?,
            in_proj_beta: make_qmatmul(in_proj_beta)?,
            in_proj_alpha: make_qmatmul(in_proj_alpha)?,
            out_proj: make_qmatmul(out_proj)?,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
        })
    }

    // L2 normalization
    fn l2_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
        let norm = (x.sqr()?.sum_keepdim(D::Minus1)? + eps as f64)?.sqrt()?;
        x.broadcast_div(&norm)
    }

    // Conv1d for prefill (multiple tokens)
    fn conv1d_prefill(&self, x: &Tensor, seq_len: usize, cache: &mut GdnLayerCache) -> Result<Tensor> {
        // x: [batch, conv_dim, seq_len]
        let pad = self.conv_kernel_size - 1;
        let device = x.device();
        let dtype = x.dtype();
        
        // Left padding
        let padding = Tensor::zeros((x.dim(0)?, self.key_dim*2 + self.value_dim, pad), dtype, device)?;
        let padded = Tensor::cat(&[padding, x.clone()], 2)?;
        
        // Save state for future steps
        cache.conv_state = padded.narrow(2, seq_len, pad)?.clone();
        
        // Apply conv1d
        let conv_out = padded.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv1d_weight.dim(0)?)?;
        candle_nn::ops::silu(&conv_out)
    }

    // Conv1d for single token generation
    fn conv1d_step(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        // x: [batch, conv_dim, 1]
        let combined = Tensor::cat(&[&cache.conv_state, x], 2)?;
        
        // Update state (keep last kernel_size-1 elements)
        cache.conv_state = combined.narrow(2, 1, self.conv_kernel_size - 1)?;
        
        // Apply conv1d
        let conv_out = combined.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv1d_weight.dim(0)?)?;
        candle_nn::ops::silu(&conv_out)
    }

    // Core delta net recurrence
    fn delta_net_recurrence(
        &self,
        q: &Tensor,      // [batch, seq, num_v_heads, head_k_dim]
        k: &Tensor,      // [batch, seq, num_v_heads, head_k_dim]
        v: &Tensor,      // [batch, seq, num_v_heads, head_v_dim]
        gate: &Tensor,   // [batch, seq, num_v_heads]
        beta: &Tensor,   // [batch, seq, num_v_heads]
        state: &mut Tensor, // [batch, num_v_heads, head_k_dim, head_v_dim]
    ) -> Result<Tensor> {
        let (batch_size, seq_len, num_heads, head_k_dim) = q.dims4()?;
        let head_v_dim = v.dim(3)?;
        
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
            let q_t = q_f32.narrow(1, t, 1)?.squeeze(1)?;
            let k_t = k_f32.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v_f32.narrow(1, t, 1)?.squeeze(1)?;
            let g_t = gate_f32.narrow(1, t, 1)?.squeeze(1)?;
            let b_t = beta_f32.narrow(1, t, 1)?.squeeze(1)?;
            
            // Apply decay: state = state * exp(g_t)
            let g_t_exp = g_t.unsqueeze(2)?.unsqueeze(3)?.exp()?;
            current_state = current_state.broadcast_mul(&g_t_exp)?;
            
            // Compute kv_mem = k_t^T @ state
            let k_t_4d = k_t.unsqueeze(3)?;
            let kv_mem = k_t_4d.transpose(2, 3)?.matmul(&current_state)?;
            let kv_mem = kv_mem.squeeze(2)?;
            
            // delta = (v_t - kv_mem) * beta_t
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&b_t.unsqueeze(2)?)?;
            
            // Update state: state = state + k_t * delta
            let delta_4d = delta.unsqueeze(3)?;
            let state_update = k_t_4d.matmul(&delta_4d.transpose(2, 3)?)?;
            current_state = (current_state + state_update)?;
            
            // Compute output: out_t = q_t @ state
            let q_t_4d = q_t.unsqueeze(2)?;
            let out_t = q_t_4d.matmul(&current_state)?.squeeze(2)?;
            outputs.push(out_t);
        }
        
        *state = current_state;
        let output = Tensor::stack(&outputs, 1)?;
        output.to_dtype(q.dtype())
    }

    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        //=========================================================================
        // 1. Input projections (quantized)
        //=========================================================================
        let original_dtype = x.dtype();
        let mut x_q = x.clone();
        if let Some(t) = self.in_proj_qkv.quantized_act_type() {
            x_q = x_q.to_dtype(t)?;
        }

        let mut qkv = MatMul.qmethod_matmul(&x_q, &*self.in_proj_qkv)?;
        let mut z = MatMul.qmethod_matmul(&x_q, &*self.in_proj_z)?;
        let mut beta_raw = MatMul.qmethod_matmul(&x_q, &*self.in_proj_beta)?;
        let mut alpha_raw = MatMul.qmethod_matmul(&x_q, &*self.in_proj_alpha)?;

        if self.in_proj_qkv.quantized_act_type().is_some() {
            qkv = qkv.to_dtype(original_dtype)?;
            z = z.to_dtype(original_dtype)?;
            beta_raw = beta_raw.to_dtype(original_dtype)?;
            alpha_raw = alpha_raw.to_dtype(original_dtype)?;
        }

        //=========================================================================
        // 2. Prepare for conv1d
        //=========================================================================
        let qkv_t = qkv.transpose(1, 2)?;  // [batch, conv_dim, seq_len]
        
        let conv_out = if seq_len == 1 && cache.seqlen_offset > 0 {
            self.conv1d_step(&qkv_t, cache)?
        } else {
            self.conv1d_prefill(&qkv_t, seq_len, cache)?
        };
        
        let conv_out = conv_out.transpose(1, 2)?;  // [batch, seq_len, conv_dim]

        //=========================================================================
        // 3. Split conv output into Q, K, V
        //=========================================================================
        let q = conv_out.narrow(2, 0, self.key_dim)?;
        let k = conv_out.narrow(2, self.key_dim, self.key_dim)?;
        let v = conv_out.narrow(2, self.key_dim * 2, self.value_dim)?;

        // Reshape to separate heads
        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        //=========================================================================
        // 4. Compute beta and gate
        //=========================================================================
        let beta = candle_nn::ops::sigmoid(&beta_raw)?;
        
        // gate = -exp(A_log) * softplus(alpha + dt_bias)
        let a_log_f32 = self.a_log.to_dtype(DType::F32)?;
        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
        let alpha_f32 = alpha_raw.to_dtype(DType::F32)?;
        
        let alpha_biased = alpha_f32.broadcast_add(&dt_bias_f32.unsqueeze(0)?.unsqueeze(0)?)?;
        let softplus = (alpha_biased.exp()? + 1.0)?.log()?;
        let neg_a_exp = a_log_f32.exp()?.neg()?;
        let gate = neg_a_exp.broadcast_mul(&softplus)?.to_dtype(dtype)?;

        //=========================================================================
        // 5. Repeat K/Q if num_v_heads != num_k_heads
        //=========================================================================
        let repeat_factor = self.num_v_heads / self.num_k_heads;
        let (q, k) = if repeat_factor > 1 {
            let q = q.unsqueeze(3)?
                .repeat((1, 1, 1, repeat_factor, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k.unsqueeze(3)?
                .repeat((1, 1, 1, repeat_factor, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        //=========================================================================
        // 6. L2 normalize Q and K
        //=========================================================================
        let q = Self::l2_norm(&q, 1e-6)?;
        let k = Self::l2_norm(&k, 1e-6)?;

        //=========================================================================
        // 7. Delta net recurrence
        //=========================================================================
        let attn_out = self.delta_net_recurrence(&q, &k, &v, &gate, &beta, &mut cache.recurrent_state)?;

        //=========================================================================
        // 8. Gated RMSNorm
        //=========================================================================
        let attn_out_flat = attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let norm_out = self.norm.forward(&attn_out_flat, &z_flat)?;
        let norm_out = norm_out.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;

        //=========================================================================
        // 9. Output projection (quantized)
        //=========================================================================
        let mut norm_out_q = norm_out;
        if let Some(t) = self.out_proj.quantized_act_type() {
            norm_out_q = norm_out_q.to_dtype(t)?;
        }
        let mut res = MatMul.qmethod_matmul(&norm_out_q, &*self.out_proj)?;
        if self.out_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }

        cache.seqlen_offset += seq_len;
        Ok(res)
    }
}


#[derive(Debug, Clone)]
pub struct QGemmaRmsNormGated {
    eps: f64,
    weight: Tensor,  // weight + 1.0
}

impl QGemmaRmsNormGated {
    pub fn new(scale: QTensor, eps: f32) -> Result<Self> {
        let weight = scale.dequantize(&scale.device())?;
        let weight = (&weight + 1.0)?;  // Gemma style
        Ok(Self {
            eps: eps as f64,
            weight,
        })
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        
        // RMS Norm
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(1)?;
        let normed = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        
        // Apply weight (already +1.0)
        let normed = normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        
        // Gate with SiLU
        let gate_f32 = gate.to_dtype(DType::F32)?;
        let gate_silu = candle_nn::ops::silu(&gate_f32)?;
        let out = normed.broadcast_mul(&gate_silu)?;
        
        out.to_dtype(x_dtype)
    }
}

//=============================================================================
// MLP with quantized weights
//=============================================================================
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
}

impl Mlp {
    fn new<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
        cfg: &Config,
    ) -> Result<Self> {
        let gate_proj = ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
        let up_proj = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
        let down_proj = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;

        let make_qmatmul = |qt: QTensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(qt),
                b: None,
            })?))
        };

        Ok(Self {
            gate_proj: make_qmatmul(gate_proj)?,
            up_proj: make_qmatmul(up_proj)?,
            down_proj: make_qmatmul(down_proj)?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let gate = MatMul.qmethod_matmul(&xs, &*self.gate_proj)?;
        let up = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let activated = crate::ops::mul_and_act(&gate, &up, self.act_fn)?;
        let mut res = MatMul.qmethod_matmul(&activated, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

//=============================================================================
// Decoder Layer
//=============================================================================

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(QGatedDeltaNet),
}

// impl LayerImpl {
//     fn forward(
//         &mut self,
//         x: &Tensor,
//         mask: Option<&Tensor>,
//         start_offsets: &[usize],
//         cache: &mut Qwen35DynamicCache,
//         layer_idx: usize,
//         metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
//         dtype: DType,
//         flash_params: &FlashParams,
//     ) -> Result<Tensor> {
//         match self {
//             LayerImpl::FullAttention(attn) => attn.forward(x, mask, start_offsets, cache, layer_idx, metadata, dtype, flash_params),
//             LayerImpl::LinearAttention(gdn) => {
//                 let gdn_cache = cache.get_gdn_cache(layer_idx)?;
//                 gdn.forward(x, gdn_cache)
//             }
//         }
//     }
// }


struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: QGemmaRmsNorm,
    post_attention_layernorm: QGemmaRmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn.forward(&x, cache)?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.mlp.forward(&normed)?;
        ffn_out + residual
    }
}

//=============================================================================
// Main Model
//=============================================================================

pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    norm: QGemmaRmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    pub cache: EitherCache,
    pub device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    num_attention_heads: usize,
    pub max_seq_len: usize,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        // Load config from GGUF metadata
        let cfg = load_config_from_gguf(&ct)?;
        let layer_types = cfg.layer_types();

        // Load embeddings
        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_embeddings, cfg.hidden_size);

        // Load final norm and output projection
        let norm = QGemmaRmsNorm::new(
            ct.tensor("output_norm.weight", device)?,
            cfg.rms_norm_eps as f32,
        )?;

        let output = if ct.has_tensor("output.weight") {
            ct.tensor("output.weight", device)?
        } else {
            ct.tensor("token_embd.weight", device)?
        };
        let lm_head: Arc<dyn QuantMethod> = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: Arc::new(output),
            b: None,
        })?);

        // Build RoPE for attention layers
        let rot_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let mut ropes = HashMap::new();
        for (i, layer_type) in layer_types.iter().enumerate().take(cfg.num_hidden_layers) {
            if matches!(layer_type, LayerType::FullAttention) {
                let rope_device = mapper.device_for(i, false).unwrap_or(device);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    ropes.entry(rope_device.location())
                {
                    let rope = RotaryEmbedding::new_partial(
                        cfg.rope_theta as f32,
                        rot_dim,
                        cfg.max_position_embeddings,
                        rope_device,
                        true,
                        DType::F32,
                    )?;
                    e.insert(Arc::new(rope));
                }
            }
        }

        // Create hybrid cache config
        let pipeline_layer_types: Vec<HybridLayerType> = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect();

        let hybrid_cache_config = HybridCacheConfig {
            layer_types: pipeline_layer_types,
            max_seq_len: cfg.max_position_embeddings,
            recurrent: RecurrentLayerConfig {
                conv_dim: cfg.linear_conv_dim(),
                conv_width: cfg.linear_conv_kernel_dim,
                state_dims: vec![
                    cfg.linear_num_value_heads,
                    cfg.linear_key_head_dim,
                    cfg.linear_value_head_dim,
                ],
            },
        };

        let pipeline_cache = Arc::new(Mutex::new(
            HybridCache::new(
                hybrid_cache_config,
                DType::F32, // TODO: get from model
                device,
            )
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to create hybrid cache: {}", e))
            })?,
        ));

        // Build layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading layers from GGUF",
            &normal_loading_metadata.multi_progress,
        ) {
            let prefix = format!("blk.{layer_idx}");
            let layer_device = mapper.device_for(layer_idx, false).unwrap_or(device);

            let layer_impl =
                match &layer_types[layer_idx] {
                    LayerType::FullAttention => {
                        let rotary_emb = ropes
                            .get(&layer_device.location())
                            .expect("No RoPE for device location!")
                            .clone();
                        let paged_attn = match &attention_mechanism {
                            AttentionImplementation::Eager => None,
                            AttentionImplementation::PagedAttention => {
                                Some(PagedAttention::new(cfg.head_dim, &layer_device, None)?)
                            }
                        };
                        LayerImpl::FullAttention(FullAttention::new(
                            &mut ct,
                            &prefix,
                            &layer_device,
                            &cfg,
                            &*mapper,
                            layer_idx,
                            rotary_emb,
                            paged_attn,
                        )?)
                    }
                    LayerType::LinearAttention => LayerImpl::LinearAttention(
                        QGatedDeltaNet::from_gguf(&mut ct, &prefix, &layer_device, &cfg)?,
                    ),
                };

            // Load layer norms
            let input_layernorm = QGemmaRmsNorm::new(
                ct.tensor(&format!("{prefix}.attn_norm.weight"), &layer_device)?,
                cfg.rms_norm_eps as f32,
            )?;
            let post_attention_layernorm = QGemmaRmsNorm::new(
                ct.tensor(
                    &format!("{prefix}.post_attention_norm.weight"),
                    &layer_device,
                )?,
                cfg.rms_norm_eps as f32,
            )?;

            // Load MLP
            let mlp = Mlp::new(&mut ct, &prefix, &layer_device, &cfg)?;

            layers.push(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                mlp,
            });
        }

        let num_attention_heads = cfg.num_attention_heads; // TODO: handle sharding

        Ok(Self {
            embed_tokens,
            layers,
            layer_types,
            norm,
            lm_head,
            cache: EitherCache::Hybrid(pipeline_cache),
            device: device.clone(),
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            num_attention_heads,
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;

        let mut hybrid_cache = self.cache.hybrid();
        let state_indices = hybrid_cache.state_indices().cloned();
        if self
            .layer_types
            .iter()
            .any(|lt| matches!(lt, LayerType::LinearAttention))
            && state_indices.is_none()
        {
            candle_core::bail!(
                "Hybrid recurrent state indices are required for linear-attention layers."
            );
        }

        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*hybrid_cache as &dyn PastKvLenCache),
            x.dtype(),
            self.num_attention_heads,
        )?;
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;

            match &layer.layer_impl {
                LayerImpl::FullAttention(_) => {
                    if let Some(HybridLayerCache::Attention(kv_cache)) =
                        hybrid_cache.get_mut(layer_idx)
                    {
                        let mask_for_layer = mask.as_ref().map(|m| m.get(x.device()).clone());
                        x = layer.forward_attention(
                            &x,
                            &mask_for_layer,
                            seqlen_offsets,
                            kv_cache,
                            metadata.as_ref().map(|(kv_cache, metadata)| {
                                (kv_cache[layer_idx].clone(), *metadata)
                            }),
                            flash_params,
                        )?;
                    }
                }
                LayerImpl::LinearAttention(_) => {
                    if let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer_idx)
                    {
                        let indices = state_indices.as_ref().expect(
                            "checked above: linear-attention layers require recurrent indices",
                        );
                        let indices_vec: Vec<u32> = indices.to_vec1()?;
                        if indices_vec.is_empty() {
                            candle_core::bail!("Hybrid recurrent state indices are empty.");
                        }

                        let first_offset = pool.get_seqlen_offset(indices_vec[0] as usize);
                        if indices_vec
                            .iter()
                            .any(|&idx| pool.get_seqlen_offset(idx as usize) != first_offset)
                        {
                            candle_core::bail!(
                                "Hybrid recurrent seqlen offsets diverged within a batch for layer {layer_idx}."
                            );
                        }

                        let conv_state = pool.gather_conv_state(indices)?;
                        let recurrent_state = pool.gather_recurrent_state(indices)?;

                        let mut gdn_cache = GdnLayerCache {
                            conv_state,
                            recurrent_state,
                            seqlen_offset: first_offset,
                        };

                        x = layer.forward_linear(&x, &mut gdn_cache)?;

                        pool.scatter_conv_state(indices, &gdn_cache.conv_state)?;
                        pool.scatter_recurrent_state(indices, &gdn_cache.recurrent_state)?;

                        let delta = gdn_cache.seqlen_offset.saturating_sub(first_offset);
                        for &idx in &indices_vec {
                            let updated = pool.get_seqlen_offset(idx as usize) + delta;
                            pool.set_seqlen_offset(idx as usize, updated);
                        }
                    } else {
                        candle_core::bail!(
                            "Hybrid cache layer {layer_idx} is not recurrent for a linear-attention layer."
                        );
                    }
                }
            }
        }

        let x = x.to_device(&self.device)?;
        let x = self.norm.forward(&x)?;

        let mut x = extract_logits(&x, context_lens)?;

        if let Some(t) = self.lm_head.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let logits = MatMul.qmethod_matmul(&x, &*self.lm_head)?;

        Ok(logits)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
            ct: Content<'_, R>,
            device: &candle_core::Device,
            mapper: Box<dyn DeviceMapper + Send + Sync>,
            attention_mechanism: AttentionImplementation,
            dtype: DType,
        ) -> anyhow::Result<Self, candle_core::Error>
        where
            Self: Sized {

    }
}

//=============================================================================
// Helper function to load config from GGUF
//=============================================================================

fn load_config_from_gguf<R: std::io::Seek + std::io::Read>(ct: &Content<'_, R>) -> Result<Config> {
    use crate::utils::gguf_metadata::TryValueInto;

    let metadata = ct.get_metadata();
    let get_val = |key: &str| -> Result<u32> {
        metadata
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg(format!("Missing metadata: {}", key)))?
            .to_u32()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse {}: {}", key, e)))
    };

    let head_count = get_val("attention.head_count")? as usize;
    let head_count_kv = get_val("attention.head_count_kv")? as usize;
    let block_count = get_val("block_count")? as usize;
    let embedding_length = get_val("embedding_length")? as usize;
    let context_length = get_val("context_length").unwrap_or(DEFAULT_MAX_SEQ_LEN) as usize;

    // Get optional values with fallbacks
    let key_length =
        get_val("attention.key_length").unwrap_or((embedding_length / head_count) as u32) as usize;
    let value_length = get_val("attention.value_length")
        .unwrap_or((embedding_length / head_count) as u32) as usize;

    let rms_norm_eps = metadata
        .get("attention.layer_norm_rms_epsilon")
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(1e-6) as f64;

    let rope_theta = metadata
        .get("rope.freq_base")
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(10_000.0) as f64;

    // GDN specific parameters
    let linear_conv_kernel_dim = metadata
        .get("ssm.conv_kernel")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4) as usize;

    let linear_key_head_dim = metadata
        .get("ssm.state_size")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(128) as usize;

    let linear_num_key_heads = metadata
        .get("ssm.group_count")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(16) as usize;

    let linear_inner_size = metadata
        .get("ssm.inner_size")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(2048) as usize;

    let linear_num_value_heads = linear_inner_size / linear_key_head_dim;

    let linear_value_head_dim = metadata
        .get("ssm.value_head_dim")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(128) as usize;

    let full_attention_interval = metadata
        .get("full_attention_interval")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4) as usize;

    Ok(Config {
        vocab_size: get_val("vocab_size")? as usize,
        hidden_size: embedding_length,
        intermediate_size: get_val("intermediate_size")? as usize,
        num_hidden_layers: block_count,
        num_attention_heads: head_count,
        num_key_value_heads: head_count_kv,
        hidden_act: crate::layers::Activation::Silu, // Qwen uses SiLU
        max_position_embeddings: context_length,
        rms_norm_eps,
        rope_theta,
        head_dim: key_length,
        partial_rotary_factor: 0.25,
        linear_conv_kernel_dim,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        full_attention_interval,
        tie_word_embeddings: true, // Qwen typically ties embeddings
        quantization_config: None,
    })
}

//=============================================================================
// Trait Implementations
//=============================================================================

impl IsqModel for ModelWeights {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));

        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    tensors.push((&mut attn.wq, Some(i)));
                    tensors.push((&mut attn.wk, Some(i)));
                    tensors.push((&mut attn.wv, Some(i)));
                    tensors.push((&mut attn.wo, Some(i)));
                }
                LayerImpl::LinearAttention(gdn) => {
                    // These are already quantized via GGUF, but we can still ISQ them
                    tensors.push((&mut gdn.in_proj_qkv, Some(i)));
                    tensors.push((&mut gdn.in_proj_z, Some(i)));
                    tensors.push((&mut gdn.in_proj_beta, Some(i)));
                    tensors.push((&mut gdn.in_proj_alpha, Some(i)));
                    tensors.push((&mut gdn.out_proj, Some(i)));
                }
            }

            // MLP layers
            tensors.push((&mut layer.mlp.gate_proj, Some(i)));
            tensors.push((&mut layer.mlp.up_proj, Some(i)));
            tensors.push((&mut layer.mlp.down_proj, Some(i)));
        }

        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        // This is for saving the model - not needed for inference
        Vec::new()
    }
}

impl NormalModel for ModelWeights {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Qwen3.5 does not support X-LoRA forward")
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn is_xlora(&self) -> bool {
        false
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for ModelWeights {}
