#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
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

/// Linear attention layer using Gated Delta Net
/// This is a simplified quantized version that uses GGUF quantized weights
struct LinearAttention {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    #[allow(dead_code)]
    conv_kernel_size: usize,

    in_proj_qkv: Arc<dyn QuantMethod>,
    in_proj_z: Arc<dyn QuantMethod>,
    in_proj_b: Arc<dyn QuantMethod>,
    in_proj_a: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,

    #[allow(dead_code)]
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm_weight: Tensor,
    norm_eps: f64,

    #[allow(dead_code)]
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl LinearAttention {
    fn new<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;

        let in_proj_qkv = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
        let in_proj_z = ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?;
        let in_proj_b = ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?;
        let in_proj_a = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?;
        let out_proj = ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?;

        let conv1d_weight = ct.tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
            .dequantize(device)?;

        let a_log = ct.tensor(&format!("{prefix}.ssm_a"), device)?.dequantize(device)?;
        let dt_bias = ct.tensor(&format!("{prefix}.ssm_dt.bias"), device)?.dequantize(device)?;
        let norm_weight = ct.tensor(&format!("{prefix}.ssm_norm.weight"), device)?.dequantize(device)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            in_proj_qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_qkv),
                b: None,
            })?),
            in_proj_z: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_z),
                b: None,
            })?),
            in_proj_b: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_b),
                b: None,
            })?),
            in_proj_a: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(in_proj_a),
                b: None,
            })?),
            out_proj: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(out_proj),
                b: None,
            })?),
            conv1d_weight,
            dt_bias,
            a_log,
            norm_weight,
            norm_eps: rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = x.dims3()?;
        let dtype = x.dtype();

        // Project input
        let mixed_qkv = MatMul.qmethod_matmul(x, &*self.in_proj_qkv)?;
        let z = MatMul.qmethod_matmul(x, &*self.in_proj_z)?;
        let b = MatMul.qmethod_matmul(x, &*self.in_proj_b)?;
        let a = MatMul.qmethod_matmul(x, &*self.in_proj_a)?;

        // Reshape
        let mixed_qkv = mixed_qkv.reshape((batch_size, seq_len, self.key_dim * 2 + self.value_dim))?;
        let z = z.reshape((batch_size, seq_len, self.value_dim))?;
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        // Split qkv
        let q = mixed_qkv.narrow(2, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(2, self.key_dim * 2, self.value_dim)?;

        // Reshape to per-head
        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // Compute beta and g
        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?;
            let a_log_f32 = self.a_log.to_dtype(DType::F32)?;
            let dt = a_f32.broadcast_add(&dt_bias_f32.unsqueeze(0)?.unsqueeze(0)?)?;
            let softplus_dt = (dt.exp()? + 1.0)?.log()?;
            let neg_a_exp = a_log_f32.exp()?.neg()?;
            neg_a_exp.broadcast_mul(&softplus_dt)?.to_dtype(dtype)?
        };

        // Repeat q, k for grouped attention
        let repeat_factor = self.num_v_heads / self.num_k_heads;
        let q = if repeat_factor > 1 {
            let q_expanded = q.unsqueeze(3)?.repeat((1, 1, 1, repeat_factor, 1))?;
            q_expanded.reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?
        } else {
            q
        };
        let k = if repeat_factor > 1 {
            let k_expanded = k.unsqueeze(3)?.repeat((1, 1, 1, repeat_factor, 1))?;
            k_expanded.reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?
        } else {
            k
        };

        // L2 normalize q and k
        let q = crate::models::gdn::l2_norm(&q, 1e-6)?;
        let k = crate::models::gdn::l2_norm(&k, 1e-6)?;

        // Apply gated delta rule recurrence
        let y = crate::models::gdn::gated_delta_rule_recurrence(
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut self.recurrent_state.clone().unwrap_or_else(|| {
                Tensor::zeros(
                    (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                    dtype,
                    x.device(),
                ).unwrap()
            }),
        )?;

        // Apply gated RMSNorm
        let y_flat = y.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let y_norm = self.rms_norm_gated(&y_flat, &z_flat)?;
        let y_norm = y_norm.reshape((batch_size, seq_len, self.value_dim))?;

        // Output projection
        MatMul.qmethod_matmul(&y_norm, &*self.out_proj)
    }

    fn rms_norm_gated(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let gate_f32 = gate.to_dtype(DType::F32)?;

        let silu_gate = candle_nn::ops::silu(&gate_f32)?;
        let variance = x_f32.sqr()?.mean_keepdim(1)?;
        let normed = x_f32.broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?;
        let out = normed.broadcast_mul(&silu_gate)?;
        let out = out.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?;
        out.to_dtype(x_dtype)
    }
}

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w1)?;
        let w3 = MatMul.qmethod_matmul(xs, &*self.feed_forward_w3)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        MatMul.qmethod_matmul(&y, &*self.feed_forward_w2)
    }
}

struct FullAttention {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        dtype: DType,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = MatMul.qmethod_matmul(x, &*self.attention_wq)?;
        let k = MatMul.qmethod_matmul(x, &*self.attention_wk)?;
        let v = MatMul.qmethod_matmul(x, &*self.attention_wv)?;

        let (q, k, v) = if seq_len != 1 {
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

        // Per-head RMSNorm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;

        let (q, k) = self.rotary.forward(&q, &k, start_offsets)?;

        let (q, k, v) = (
            q.to_dtype(dtype)?,
            k.to_dtype(dtype)?,
            v.to_dtype(dtype)?,
        );

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

        let y = if mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        MatMul.qmethod_matmul(&y.to_dtype(x.dtype())?, &*self.attention_wo)
    }
}

enum TokenMixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

struct LayerWeights {
    token_mixer: TokenMixer,
    mlp: Mlp,
    input_layernorm: QRmsNorm,
    post_attention_layernorm: QRmsNorm,
}

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
}

fn verify_qwen35_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen35" {
        candle_core::bail!("Expected `qwen35` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen35_arch(c.metadata)?;

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

        // Helper to get metadata with fallback
        let get_meta = |key: &str, alt_key: &str| -> Option<u32> {
            c.get_value::<u32>(key).ok().or_else(|| c.get_value::<u32>(alt_key).ok())
        };

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
            full_attention_interval: get_meta("full_attention_interval", "qwen3.full_attention_interval")
                .unwrap_or(4) as usize,
            linear_num_key_heads: get_meta("ssm.group_count", "qwen3.ssm.group_count")
                .unwrap_or(4) as usize,
            linear_num_value_heads: get_meta("ssm.num_value_heads", "qwen3.ssm.num_value_heads")
                .unwrap_or(8) as usize,
            linear_key_head_dim: get_meta("ssm.state_size", "qwen3.ssm.state_size")
                .unwrap_or(128) as usize,
            linear_value_head_dim: get_meta("ssm.value_head_dim", "qwen3.ssm.value_head_dim")
                .unwrap_or(128) as usize,
            linear_conv_kernel_dim: get_meta("ssm.conv_kernel", "qwen3.ssm.conv_kernel")
                .unwrap_or(4) as usize,
        };

        Ok(props)
    }
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

        ct.print_metadata()?;

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
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };

        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    device,
                    true,
                    DType::F32,
                )?),
            );
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rotary = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();

            let is_full_attention = (layer_idx + 1) % full_attention_interval == 0;

            let token_mixer = if is_full_attention {
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

                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(head_dim, device, None)?)
                    }
                };

                TokenMixer::Full(FullAttention {
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
                    rotary: rotary.clone(),
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: head_count / head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                        sliding_window: None,
                        sinks: None,
                    },
                })
            } else {
                TokenMixer::Linear(LinearAttention::new(
                    &mut ct,
                    &prefix,
                    device,
                    linear_num_value_heads,
                    linear_num_key_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    linear_conv_kernel_dim,
                    rms_norm_eps as f64,
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

            let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?;

            layers.push(LayerWeights {
                token_mixer,
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

        let n_head = match &self.layers[0].token_mixer {
            TokenMixer::Full(ref attn) => attn.n_head,
            TokenMixer::Linear(ref linear) => linear.num_v_heads,
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
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }

            let x = layer_in;
            let residual = &x;
            let x = layer.input_layernorm.forward(&x)?;

            let attn = match &mut layer.token_mixer {
                TokenMixer::Full(ref mut attn) => attn.forward(
                    &x,
                    mask.as_ref().map(|m| m.get(x.device())),
                    start_offsets,
                    &mut cache[i],
                    metadata
                        .as_ref()
                        .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    self.dtype,
                )?,
                TokenMixer::Linear(ref mut linear) => linear.forward(&x)?,
            };

            let x = (attn + residual)?;

            let residual = &x;
            let x = layer.post_attention_layernorm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            layer_in = (x + residual)?;
        }

        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        MatMul.qmethod_matmul(&x.contiguous()?, &*self.output)
    }
}
