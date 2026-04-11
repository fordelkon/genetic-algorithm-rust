use rand::Rng;

use crate::ga::error::GaError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// 单个基因(gene)的标量类型定义
pub enum GeneScalarType {
    Isize,
    I8,
    I16,
    I32,
    I64,
    Usize,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Float16,
    Object,
}

#[derive(Debug, Clone, PartialEq)]
/// 单个基因(gene)的真实值表示
pub enum GeneValue {
    Isize(isize),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Usize(usize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone)]
/// 染色体(genes)值类型，全部统一/按基因单独定义
pub enum GenesValueType {
    All(GeneScalarType),
    PerGene(Vec<GeneScalarType>),
}

#[derive(Debug, Clone)]
/// 基因(gene)取值空间，离散域/连续域/等距离散域
pub enum GeneDomain {
    Discrete(Vec<f64>),
    Continuous { low: f64, high: f64 },
    Stepped { low: f64, high: f64, step: f64 },
}

#[derive(Debug, Clone)]
/// 染色体(genes)取值空间，全局统一/按基因单独定义
pub enum GenesDomain {
    Global(GeneDomain),
    PerGene(Vec<GeneDomain>),
}

impl GeneScalarType {
    /// 返回该类型是否是当前版本支持的数值类型
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Float16 | Self::Object)
    }

    /// 返回类型名称
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Isize => "isize",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::Usize => "usize",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Float16 => "float16",
            Self::Object => "object",
        }
    }

    pub fn is_unsigned(self) -> bool {
        matches!(
            self,
            Self::Usize | Self::U8 | Self::U16 | Self::U32 | Self::U64
        )
    }
}

impl GeneValue {
    /// 返回基因值对应的标量类型
    pub fn scalar_type(&self) -> GeneScalarType {
        match self {
            Self::Isize(_) => GeneScalarType::Isize,
            Self::I8(_) => GeneScalarType::I8,
            Self::I16(_) => GeneScalarType::I16,
            Self::I32(_) => GeneScalarType::I32,
            Self::I64(_) => GeneScalarType::I64,
            Self::Usize(_) => GeneScalarType::Usize,
            Self::U8(_) => GeneScalarType::U8,
            Self::U16(_) => GeneScalarType::U16,
            Self::U32(_) => GeneScalarType::U32,
            Self::U64(_) => GeneScalarType::U64,
            Self::F32(_) => GeneScalarType::F32,
            Self::F64(_) => GeneScalarType::F64,
        }
    }

    /// 统一转换为 `f64`，用于 fitness 计算、比较和统计
    pub fn to_f64(&self) -> f64 {
        match self {
            Self::Isize(value) => *value as f64,
            Self::I8(value) => *value as f64,
            Self::I16(value) => *value as f64,
            Self::I32(value) => *value as f64,
            Self::I64(value) => *value as f64,
            Self::Usize(value) => *value as f64,
            Self::U8(value) => *value as f64,
            Self::U16(value) => *value as f64,
            Self::U32(value) => *value as f64,
            Self::U64(value) => *value as f64,
            Self::F32(value) => *value as f64,
            Self::F64(value) => *value,
        }
    }

    /// 将一个 `f64` 映射为指定标量类型的真实基因值
    pub fn cast_from_f64(scalar_type: GeneScalarType, value: f64) -> Result<Self, GaError> {
        match scalar_type {
            GeneScalarType::Isize => clamp_rounded(value, isize::MIN as f64, isize::MAX as f64)
                .map(|value| Self::Isize(value as isize)),
            GeneScalarType::I8 => clamp_rounded(value, i8::MIN as f64, i8::MAX as f64)
                .map(|value| Self::I8(value as i8)),
            GeneScalarType::I16 => clamp_rounded(value, i16::MIN as f64, i16::MAX as f64)
                .map(|value| Self::I16(value as i16)),
            GeneScalarType::I32 => clamp_rounded(value, i32::MIN as f64, i32::MAX as f64)
                .map(|value| Self::I32(value as i32)),
            GeneScalarType::I64 => clamp_rounded(value, i64::MIN as f64, i64::MAX as f64)
                .map(|value| Self::I64(value as i64)),
            GeneScalarType::Usize => clamp_rounded(value, 0.0, usize::MAX as f64)
                .map(|value| Self::Usize(value as usize)),
            GeneScalarType::U8 => {
                clamp_rounded(value, 0.0, u8::MAX as f64).map(|value| Self::U8(value as u8))
            }
            GeneScalarType::U16 => {
                clamp_rounded(value, 0.0, u16::MAX as f64).map(|value| Self::U16(value as u16))
            }
            GeneScalarType::U32 => {
                clamp_rounded(value, 0.0, u32::MAX as f64).map(|value| Self::U32(value as u32))
            }
            GeneScalarType::U64 => {
                clamp_rounded(value, 0.0, u64::MAX as f64).map(|value| Self::U64(value as u64))
            }
            GeneScalarType::F32 => Ok(Self::F32(value as f32)),
            GeneScalarType::F64 => Ok(Self::F64(value)),
            GeneScalarType::Float16 | GeneScalarType::Object => {
                Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()))
            }
        }
    }
}

impl GenesValueType {
    /// 返回染色体(genes)中第`gene_index`个基因的值类型
    pub fn value_type_for(&self, gene_index: usize) -> GeneScalarType {
        match self {
            Self::All(value_type) => *value_type,
            Self::PerGene(types) => types[gene_index],
        }
    }
}

impl GenesDomain {
    /// 返回染色体(genes)中第`gene_index`个基因的取值空间
    pub fn domain_for(&self, gene_index: usize) -> &GeneDomain {
        match self {
            Self::Global(domain) => domain,
            Self::PerGene(domains) => &domains[gene_index],
        }
    }
}

impl GeneDomain {
    /// 校检基因(gene)取值空间的合法性
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Discrete(values) if values.is_empty() => {
                Err("discrete gene domain must not be empty".into())
            }
            Self::Continuous { low, high } if low > high => {
                Err("continuous gene domain requires low <= high".into())
            }
            Self::Stepped { low, high, .. } if low > high => {
                Err("stepped gene domain requires low <= high".into())
            }
            Self::Stepped { step, .. } if *step <= 0.0 => {
                Err("stepped gene domain requires step > 0".into())
            }
            _ => Ok(()),
        }
    }

    /// 检查取值空间是否与目标标量类型兼容
    pub fn validate_for_type(&self, scalar_type: GeneScalarType) -> Result<(), GaError> {
        if !scalar_type.is_supported() {
            return Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()));
        }

        match self {
            Self::Discrete(values) => {
                for value in values {
                    let normalized = if scalar_type.is_unsigned() && *value < 0.0 {
                        return Err(GaError::InvalidConfig(format!(
                            "discrete gene domain value {value} cannot be represented by {}",
                            scalar_type.as_str()
                        )));
                    } else {
                        *value
                    };
                    let _ = GeneValue::cast_from_f64(scalar_type, normalized)?;
                }
                Ok(())
            }
            Self::Continuous { low, high } => {
                if scalar_type.is_unsigned() && *low < 0.0 {
                    return Err(GaError::InvalidConfig(format!(
                        "continuous gene domain low {low} cannot be represented by {}",
                        scalar_type.as_str()
                    )));
                }
                let _ = GeneValue::cast_from_f64(scalar_type, *low)?;
                let _ = GeneValue::cast_from_f64(scalar_type, *high)?;
                Ok(())
            }
            Self::Stepped { low, high, step } => {
                if scalar_type.is_unsigned() && *low < 0.0 {
                    return Err(GaError::InvalidConfig(format!(
                        "stepped gene domain low {low} cannot be represented by {}",
                        scalar_type.as_str()
                    )));
                }
                let _ = GeneValue::cast_from_f64(scalar_type, *low)?;
                let _ = GeneValue::cast_from_f64(scalar_type, *high)?;
                let _ = GeneValue::cast_from_f64(scalar_type, *step)?;
                Ok(())
            }
        }
    }

    /// 返回基因(gene)取值空间中随机采样的数值表示
    pub fn sample_numeric(&self, rng: &mut impl Rng) -> f64 {
        match self {
            Self::Discrete(values) => {
                let index = rng.gen_range(0..values.len());
                values[index]
            }
            Self::Continuous { low, high } => rng.gen_range(*low..=*high),
            Self::Stepped { low, high, step } => {
                let steps = ((*high - *low) / *step).floor() as usize;
                let offset = rng.gen_range(0..=steps);
                *low + *step * offset as f64
            }
        }
    }

    /// 返回`value`映射到基因(gene)取值空间中的数值表示
    pub fn normalize_numeric(&self, value: f64) -> f64 {
        match self {
            Self::Discrete(values) => *values
                .iter()
                .min_by(|left, right| {
                    (value - **left)
                        .abs()
                        .partial_cmp(&(value - **right).abs())
                        .expect("distance comparison failed")
                })
                .expect("discrete gene domain must not be empty"),
            Self::Continuous { low, high } => value.clamp(*low, *high),
            Self::Stepped { low, high, step } => {
                let clamped = value.clamp(*low, *high);
                let offset = ((clamped - *low) / *step).round();
                let snapped = *low + offset * *step;
                snapped.clamp(*low, *high)
            }
        }
    }
}

fn clamp_rounded(value: f64, min: f64, max: f64) -> Result<f64, GaError> {
    if !value.is_finite() {
        return Err(GaError::InvalidConfig("gene value must be finite".into()));
    }

    Ok(value.round().clamp(min, max))
}
