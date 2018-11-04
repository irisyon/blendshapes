#define DIFFUSE_PREMULTIPLIED 
#define HAS_NORMAL 
#define LOCK_AMBIENT_WITH_DIFFUSE 
#define METAL 1
#define USE_DIFFUSE 
#define USE_DIFFUSE_COLOR 
#define USE_MODELVIEWPROJECTIONTRANSFORM 
#define USE_NODE_OPACITY 
#define kSCNTexcoordCount 0
////////////////////////////////////////////////
// CommonProfile Shader v2

#import <metal_stdlib>

using namespace metal;

#ifndef __SCNMetalDefines__
#define __SCNMetalDefines__

enum {
    SCNVertexSemanticPosition,
    SCNVertexSemanticNormal,
    SCNVertexSemanticTangent,
    SCNVertexSemanticColor,
    SCNVertexSemanticBoneIndices,
    SCNVertexSemanticBoneWeights,
    SCNVertexSemanticTexcoord0,
    SCNVertexSemanticTexcoord1,
    SCNVertexSemanticTexcoord2,
    SCNVertexSemanticTexcoord3,
    SCNVertexSemanticTexcoord4,
    SCNVertexSemanticTexcoord5,
    SCNVertexSemanticTexcoord6,
    SCNVertexSemanticTexcoord7
};

// This structure hold all the informations that are constant through a render pass
// In a shader modifier, it is given both in vertex and fragment stage through an argument named "scn_frame".
struct SCNSceneBuffer {
    float4x4    viewTransform;
    float4x4    inverseViewTransform; // transform from view space to world space
    float4x4    projectionTransform;
    float4x4    viewProjectionTransform;
    float4x4    viewToCubeTransform; // transform from view space to cube texture space (canonical Y Up space)
    float4x4    lastFrameViewProjectionTransform;
    float4      ambientLightingColor;
    float4		fogColor;
    float3		fogParameters; // x:-1/(end-start) y:1-start*x z:exp
    float2      inverseResolution;
    float       time;
    float       sinTime;
    float       cosTime;
    float       random01;
    float       motionBlurIntensity;
    // new in macOS 10.12 and iOS 10
    float       environmentIntensity;
    float4x4    inverseProjectionTransform;
    float4x4    inverseViewProjectionTransform;
    // new in macOS 10.13 and iOS 11
    float2      nearFar; // x: near, y: far
    float4      viewportSize; // xy:size, zw:origin
    // new in macOS 10.14 and iOS 12
    float4x4    inverseTransposeViewTransform;

    // internal, DO NOT USE
    float4      clusterScale; // w contains z bias
};

// In custom shaders or in shader modifiers, you also have access to node relative information.
// This is done using an argument named "scn_node", which must be a struct with only the necessary fields
// among the following list:
//
// float4x4 modelTransform;
// float4x4 inverseModelTransform;
// float4x4 modelViewTransform;
// float4x4 inverseModelViewTransform;
// float4x4 normalTransform; // This is the inverseTransposeModelViewTransform, need for normal transformation
// float4x4 modelViewProjectionTransform;
// float4x4 inverseModelViewProjectionTransform;
// float2x3 boundingBox;
// float2x3 worldBoundingBox;

#endif /* defined(__SCNMetalDefines__) */


//
// Utility
//

// Helper for compute kernels
#ifndef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
    #define RETURN_IF_OUTSIDE(dst) if ((index.x >= dst.get_width()) || (index.y >= dst.get_height())) return;
    #define RETURN_IF_OUTSIDE3D(dst) if ((index.x >= dst.get_width()) || (index.y >= dst.get_height()) || (index.z >= dst.get_depth())) return;
#else // On macOS, we use dispatchThread with won't execute on out of texture pixels
    #define RETURN_IF_OUTSIDE(dst)
    #define RETURN_IF_OUTSIDE3D(dst)
#endif

// Tool function

namespace scn {
    
    // MARK: - Matrix/Vector utils
    
    inline float3x3 mat3(float4x4 mat4)
    {
        return float3x3(mat4[0].xyz, mat4[1].xyz, mat4[2].xyz);
    }
    
    inline float3 mat4_mult_float3_normalized(float4x4 matrix, float3 src)
    {
        float3 dst  =  src.xxx * matrix[0].xyz;
        dst         += src.yyy * matrix[1].xyz;
        dst         += src.zzz * matrix[2].xyz;
        return normalize(dst);
    }
    
    inline float3 mat4_mult_float3(float4x4 matrix, float3 src)
    {
        float3 dst  =  src.xxx * matrix[0].xyz;
        dst         += src.yyy * matrix[1].xyz;
        dst         += src.zzz * matrix[2].xyz;
        return dst;
    }

    inline float3 matrix_rotate(float4x4 mat, float3 dir)
    {
        return  dir.xxx * mat[0].xyz +
                dir.yyy * mat[1].xyz +
                dir.zzz * mat[2].xyz;
    }

    inline float4 matrix_transform(float4x4 mat, float3 pos)
    {
        return  pos.xxxx * mat[0] +
                pos.yyyy * mat[1] +
                pos.zzzz * mat[2] +
                           mat[3];
    }

    inline float3 quaternion_rotate_vector(float4 q, float3 v)
    {
        float3 t = 2.f * cross(q.xyz, v);
        return v + q.w * t + cross(q.xyz, t);
    }

    // This seems unneeded with float. Maybe half ?
    template <class T>
    inline vec<T, 3> robust_normalize(vec<T, 3> v)
    {
        vec<T, 3> zero = 0.;
        return all(v == zero) ? zero : normalize(v);
    }

    template <class T>
    inline void generate_basis(vec<T, 3> inR, thread vec<T, 3> *outS, thread vec<T, 3> *outT)
    {
        // from http://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html
        T x  = -inR.x;
        T y  = inR.y;
        T z  = inR.z;
        T sz = copysign(T(1.), z);
        T a  = y / (abs(z) + T(1.));
        T b  = y * a;
        T c  = x * a;
        *outS = (vec<T, 3>){ z + sz * b,  sz * c,       x       };
        *outT = (vec<T, 3>){ c,           T(1.) - b,    -sz * y };
    }
    
    // MARK: - Blending operators
    
    inline float3 blend_add(float3 base, float3 blend)
    {
        return min(base + blend, 1.0);
    }
    
    inline float3 blend_lighten(float3 base, float3 blend)
    {
        return max(blend, base);
    }
    
    inline float3 blend_screen(float3 base, float3 blend)
    {
        return (1.0 - ((1.0 - base) * (1.0 - blend)));
    }

    // MARK: - Math
    
    inline half sq(half f) {
        return f * f;
    }

    inline float sq(float f) {
        return f * f;
    }
    
    inline float2 sincos(float angle) {
        float cs;
        float sn = ::sincos(angle, cs);
        return float2(sn, cs);
    }
    
    // max error ~ 9.10-3
    inline float acos_fast(float f) {
        float x = abs(f);
        float res = -0.156583f * x + M_PI_2_F;
        res *= sqrt(1.0f - x);
        return (f >= 0.f) ? res : M_PI_F - res;
    }

    inline float asin_fast(float f)
    {
        return M_PI_2_F - acos_fast(f);
    }

    // From Michal Drobot
    inline float atan_fast(float inX)
    {
        float  x = inX;
        return x*(-0.1784f * abs(x) - 0.0663f * x * x + 1.0301f);
    }
    
    inline float atan2_fast(float y, float x)
    {
        float sx = x > 0.f ? -1.f : 1.f;
        float abs_y = abs(y) + 1e-10f; // epsilon to prevent 0/0 condition
        float r = (x + abs_y*sx) / (abs_y - x*sx);
        float angle = sx * M_PI_4_F + M_PI_2_F;
        angle      += (0.1963f * r * r - 0.9817f) * r;
        return y > 0.f ? angle : -angle;
    }
    
    // phi/theta are in the [0..1] range
    template <class T>
    inline vec<T, 3> cartesian_from_spherical(vec<T, 2> uv)
    {
        // do not use sinpi() waiting for
        // <rdar://problem/28486742> sinpi(x) is 3x slower than sin(x * PI) on N71
        T cos_phi;
        T phi = uv.x * 2.0f * M_PI_F;
        T sin_phi = ::sincos(phi, cos_phi);
        
        T cos_theta;
        T theta     = uv.y * M_PI_F;
        T sin_theta = ::sincos(theta, cos_theta);

        return (vec<T, 3>)(cos_phi * sin_theta,
                           cos_theta,
                           -sin_phi * sin_theta);
    }

    inline float2 spherical_from_cartesian(float3 dir)
    {
        return float2( atan2(-dir.z, dir.x) * (0.5f * M_1_PI_F), acos(dir.y) * M_1_PI_F);
    }

    inline half2 spherical_from_cartesian(half3 dir)
    {
        return half2(atan2(-dir.z, dir.x) * 0.5h, acos(dir.y)) * M_1_PI_H;
    }

    inline float2 spherical_from_cartesian_fast(float3 dir)
    {
        return float2( atan2_fast(-dir.z, dir.x) * (0.5f * M_1_PI_F), acos_fast(dir.y) * M_1_PI_F);
    }

    inline half2 spherical_from_cartesian_fast(half3 dir)
    {
        return half2( atan2_fast(-dir.z, dir.x) * 0.5h, acos_fast(dir.y)) * M_1_PI_H;
    }

    #define dual_contract_factor  1.0

    template <class T>
    inline vec<T, 2> dual_paraboloid_from_cartesian(vec<T, 3> dir)
    {
        dir.xy /= abs(dir.z) + 1.0;
//        dir.xy /= dual_contract_factor;
        dir.y = 0.5 - dir.y * 0.5;
        T s   = sign(dir.z) * 0.25;
        dir.x = s * (dir.x - 1.0) + 0.5;
        return dir.xy;
    }
    
    // uv [0..1]
    template <class T>
    inline vec<T, 3> cartesian_from_dual_paraboloid(vec<T, 2>  uv)
    {
        // put uv in [-1..1] for each side
        T zside = 0.5 * sign(0.5 - uv.x);
        uv.x = 1.0 - abs(4.0 * uv.x - 2.0); // [-1..1|1..-1]
        uv.y   = 1.0 - uv.y * 2.0;
        T z = length_squared(uv); // * T(dual_contract_factor);
        z = (1.0 - z) * zside;
        
        return vec<T, 3>(uv.x, uv.y, z);
    }

    inline float reduce_min(float3 v) {
        return min(v.x, min(v.y, v.z));
    }
    
    inline float reduce_min(float4 v) {
        return min(min(v.x, v.y), min(v.z, v.w));
    }

    inline float reduce_max(float3 v) {
        return max(v.x, max(v.y, v.z));
    }

    inline float reduce_max(float4 v) {
        return max(max(v.x, v.y), max(v.z, v.w));
    }
    
    inline float3 randomSphereDir(float2 rnd)
    {
        float s = rnd.x * M_PI_F * 2.f;
        float t = rnd.y * 2.f - 1.f;
        return float3(sin(s), cos(s), t) / sqrt(1.f + t * t);
    }
    
    // from Sledgehammer slides
    template <class T>
    inline T interleaved_gradient_noise(vec<T, 2> pos)
    {
        vec<T, 3> magic( 0.06711056f, 0.00583715f, 52.9829189f );
        return fract( magic.z * fract( dot( pos, magic.xy ) ) );
    }
    
    inline float3 hemisphere_reflect(float3 v, float3 nrm)
    {
        return v * sign(dot(v, nrm));
    }
    
    inline float3 randomHemisphereDir(float3 dir, float2 rnd)
    {
        return hemisphere_reflect(randomSphereDir( rnd ), dir);
    }
    
    inline void orthogonal_basis(float3 n, thread float3& xp, thread float3& yp)
    {
        // method 2a variant
        float sz = n.z >= 0.f ? 1.f : -1.f;
        float a  =  n.y / (1.f + abs(n.z));
        float b  =  n.y * a;
        float c  = -n.x * a;
        
        xp = float3(n.z + sz * b,   sz * c,     -n.x);
        yp = float3(c,              1.f - b,    -sz * n.y);
    }

    template <class U>
    inline float2 normalized_coordinate(ushort2 index, U texture)
    {
        return float2(float(index.x) / float(texture.get_width()  - 1),
                      float(index.y) / float(texture.get_height() - 1));
    }

    template <class U>
    inline float2 normalized_coordinate(uint2 index, U texture)
    {
        return float2(float(index.x) / float(texture.get_width()  - 1),
                      float(index.y) / float(texture.get_height() - 1));
    }

    template <class U>
    inline half2 normalized_coordinate_half(uint2 index, U texture)
    {
        return half2(half(index.x) / half(texture.get_width()  - 1),
                     half(index.y) / half(texture.get_height() - 1));
    }

    // MARK: Working with cube textures

    template <class T>
    inline vec<T, 3> cubemap_dir_from_sampleCoord(uint face, vec<T, 2> sampleCoord) // sampleCoord in [-1, 1]
    {
        switch(face) {
            case 0: // +X
                return vec<T, 3>( 1.0, -sampleCoord.y, -sampleCoord.x);

            case 1: // -X
                return vec<T, 3>(-1.0, -sampleCoord.y,  sampleCoord.x);

            case 2: // +Y
                return vec<T, 3>(sampleCoord.x,  1.0,  sampleCoord.y);

            case 3: // -Y
                return vec<T, 3>(sampleCoord.x, -1.0, -sampleCoord.y);

            case 4: // +Z
                return vec<T, 3>( sampleCoord.x, -sampleCoord.y,  1.0);

            default: // -Z
                return vec<T, 3>(-sampleCoord.x, -sampleCoord.y, -1.0);
        }
    }

    // convert form [0..1] to [-1..1]
    template <class T>
    inline T signed_unit(T uv) {
        return uv * 2.0 - 1.0;
    }

    // convert form [-1..1] to [0..1]
    template <class T>
    inline T unsigned_unit(T uv) {
        return uv * 0.5 + 0.5;
    }

    template <class T>
    inline vec<T, 3> cubemap_dir_from_uv(uint face, vec<T, 2> uv) // uv in [0, 1]
    {
        return cubemap_dir_from_sampleCoord(face, signed_unit(uv));
    }

    template <class T>
    inline vec<T, 3> cubemap_dir_from_uv_unit(uint face, vec<T, 2> uv) // uv in [0, 1]
    {
        return normalize(cubemap_dir_from_uv(face, uv));
    }

    // MARK: - SIMD Extensions
    
    inline vector_float2 barycentric_mix(vector_float2 __x, vector_float2 __y, vector_float2 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    inline vector_float3 barycentric_mix(vector_float3 __x, vector_float3 __y, vector_float3 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    inline vector_float4 barycentric_mix(vector_float4 __x, vector_float4 __y, vector_float4 __z, vector_float3 __t) { return __t.x * __x + __t.y * __y + __t.z * __z; }
    
    static inline float rect(float2 lt, float2 rb, float2 uv)
    {
        float2 borders = step(lt, uv) * step(uv, rb);
        return borders.x * borders.y;
    }
    
    inline half4 debugColorForCascade(int cascade)
    {
        switch (cascade) {
            case 0:
            return half4(1.h, 0.h, 0.h, 1.h);
            case 1:
            return half4(0.9, 0.5, 0., 1.);
            case 2:
            return half4(1., 1., 0., 1.);
            case 3:
            return half4(0., 1., 0., 1.);
            default:
            return half4(0., 0., 0., 1.);
        }
    }

    inline half3 debugColorForFace(int count)
    {
        switch (count) {
            case 0:  return half3(1.0h, 0.1h, 0.1h);
            case 1:  return half3(0.1h, 1.0h, 1.0h);
            case 2:  return half3(0.1h, 1.0h, 0.1h);
            case 3:  return half3(1.0h, 0.1h, 1.0h);
            case 4:  return half3(0.1h, 0.1h, 1.0h);
            default: return half3(1.0h, 1.0h, 0.1h);
        }
    }

    inline half4 debugColorForCount(int count)
    {
        switch (count) {
            case 0: return half4(0.0h, 0.0h, 0.0h, 1.h);
            case 1: return half4(0.0h, 0.0h, 0.4h, 1.h);
            case 2: return half4(0.0h, 0.0h, 0.9h, 1.h);
            case 3: return half4(0.0h, 0.4h, 0.7h, 1.h);
            case 4: return half4(0.0h, 0.9h, 0.4h, 1.h);
            case 5: return half4(0.0h, 0.9h, 0.0h, 1.h);
            case 6: return half4(0.4h, 0.7h, 0.0h, 1.h);
            case 7: return half4(0.9h, 0.7h, 0.0h, 1.h);
            default: return half4(1., 0., 0., 1.);
        }
    }

    inline float grid(float2 lt, float2 rb, float2 gridSize, float thickness, float2 uv)
    {
        float insideRect = rect(lt, rb + thickness, uv);
        float2 gt = thickness * gridSize;
        float2 lines = step(abs(lt - fract(uv * gridSize)), gt);
        return insideRect * (lines.x + lines.y);
    }

    inline float checkerboard(float2 gridSize, float2 uv)
    {
        float2 check = floor(uv * gridSize);
        return step(fmod(check.x + check.y, 2.f), 0.f);
    }

    // MARK: - Colors
    
    inline float luminance(float3 color)
    {
        // `color` assumed to be in the linear sRGB color space
        // https://en.wikipedia.org/wiki/Relative_luminance
        return color.r * 0.212671 + color.g * 0.715160 + color.b * 0.072169;
    }
    
    inline float srgb_to_linear(float c)
    {
        return (c <= 0.04045f) ? c / 12.92f : powr((c + 0.055f) / 1.055f, 2.4f);
    }
    
    inline half srgb_to_linear_fast(half c)
    {
        return powr(c, 2.2h);
    }
    
    inline half3 srgb_to_linear_fast(half3 c)
    {
        return powr(c, 2.2h);
    }
    
    inline half srgb_to_linear(half c)
    {
        // return (c <= 0.04045h) ? c / 12.92h : powr((c + 0.055h) / 1.055h, 2.4h);
        return (c <= 0.04045h) ? (c * 0.0773993808h) :  powr(0.9478672986h * c + 0.05213270142h, 2.4h);
    }
    
    inline float3 srgb_to_linear(float3 c)
    {
        return float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
    }
    
    inline float linear_to_srgb(float c)
    {
        return (c < 0.0031308f) ? (12.92f * c) : (1.055f * powr(c, 1.f/2.4f) - 0.055f);
    }
    
    inline float3 linear_to_srgb(float3 v) { // we do not saturate since linear extended values can be fed in
        return float3(linear_to_srgb(v.x), linear_to_srgb(v.y), linear_to_srgb(v.z));
    }
    
}

// MARK: GL helpers

template <typename T>
inline T dFdx(T v) {
    return dfdx(v);
}

// Y is up in GL and down in Metal
template <typename T>
inline T dFdy(T v) {
    return -dfdy(v);
}

// MARK: -

inline float4 texture2DProj(texture2d<float> tex, sampler smp, float4 uv)
{
    return tex.sample(smp, uv.xy / uv.w);
}

inline half4 texture2DProj(texture2d<half> tex, sampler smp, float4 uv)
{
    return tex.sample(smp, uv.xy / uv.w);
}

static constexpr sampler scn_shadow_sampler_rev_z = sampler(coord::normalized, filter::linear, mip_filter::none, address::clamp_to_zero, compare_func::less_equal);
static constexpr sampler scn_shadow_sampler_ord_z = sampler(coord::normalized, filter::linear, mip_filter::none, address::clamp_to_edge, compare_func::greater_equal);

#if defined(USE_REVERSE_Z) && USE_REVERSE_Z
static constexpr sampler scn_shadow_sampler = scn_shadow_sampler_rev_z;
#else
static constexpr sampler scn_shadow_sampler = scn_shadow_sampler_ord_z;
#endif

inline float shadow2DProj(sampler shadow_sampler, depth2d<float> tex, float4 uv)
{
    float3 uvp = uv.xyz / uv.w;
    return tex.sample_compare(shadow_sampler, uvp.xy, uvp.z);
}

inline float shadow2DArray(sampler shadow_sampler, depth2d_array<float> tex, float3 uv, uint slice)
{
    return tex.sample_compare(shadow_sampler, uv.xy, slice, uv.z);
}

inline float shadow2DArrayProj(sampler shadow_sampler, depth2d_array<float> tex, float4 uv, uint slice)
{
    float3 uvp = uv.xyz / uv.w;
    return tex.sample_compare(shadow_sampler, uvp.xy, slice, uvp.z);
}

// MARK Shadow

inline float4 transformViewPosInShadowSpace(float3 pos, float4x4 shadowMatrix, bool reverseZ)
{
    //project into light space
    float4 lightScreen =  shadowMatrix * float4(pos, 1.f);
    
    // ensure receiver after the shadow projection box are not in shadow (when no caster == 1. instead of infinite)
    // TODO : this is awkward : we maybe should rework the comparison order to have something more natural
    lightScreen.z = reverseZ ? max(lightScreen.z, 0.f) : min(lightScreen.z, 0.9999f * lightScreen.w);
    
    return lightScreen;
}

inline float ComputeShadow(sampler shadow_sampler, float3 worldPos, float4x4 shadowMatrix, depth2d<float> shadowMap, bool reverseZ)
{
    float4 lightScreen =  transformViewPosInShadowSpace(worldPos, shadowMatrix, reverseZ);

    float shadow = shadow2DProj(shadow_sampler, shadowMap, lightScreen);

    // Is this useful ?
    shadow *= step(0., lightScreen.w);
    
    return shadow;
}

inline float ComputeSoftShadowGrid(sampler shadow_sampler, float3 worldPos, float4x4 shadowMatrix, depth2d<float> shadowMap, int sampleCount, bool reverseZ)
{
    float4 lightScreen =  transformViewPosInShadowSpace(worldPos, shadowMatrix, reverseZ);

    // if sampleCount is known compileTime, this get rid of the shadowKernel binding
    float shadow;
    if (sampleCount <= 1) {
        shadow = shadow2DProj(shadow_sampler, shadowMap, lightScreen);
    } else {

        float3 uvp = lightScreen.xyz / lightScreen.w;
        uvp.z += reverseZ ? 0.005f : -0.005f; // TODO: get rid of hardcoded bias...

        float2 texelSize = 2.f / float2(shadowMap.get_width(), shadowMap.get_height());
        float2 origin    = uvp.xy - (sampleCount * 0.5f) * texelSize;

        // penumbra
        if (sampleCount <= 4) { // offset are limited to [-7..8]
            half totalAccum = 0.h;
            for (int y = 0; y < sampleCount; ++y) {
                for (int x = 0; x < sampleCount; ++x) {
                    totalAccum  += half(shadowMap.sample_compare(shadow_sampler, origin, uvp.z, 2 * int2(x,y)));
                }
            }
            shadow = totalAccum / half(sampleCount * sampleCount);
        } else {
            float totalAccum = 0.f;
            for (int y = 0; y < sampleCount; ++y) {
                for (int x = 0; x < sampleCount; ++x) {
                    float2 samplePos = origin + texelSize * float2(x, y);
                    totalAccum  += shadowMap.sample_compare(shadow_sampler, samplePos, uvp.z);
                }
            }
            shadow = totalAccum / float(sampleCount * sampleCount);
        }
    }

    // Is this useful ?
    shadow *= step(0., lightScreen.w);

    return shadow;
}

inline float ComputeSoftShadow(sampler shadow_sampler, float3 worldPos, float4x4 shadowMatrix, depth2d<float> shadowMap, constant float4* shadowKernel, int sampleCount, float shadowRadius, bool reverseZ)
{
    float4 lightScreen =  transformViewPosInShadowSpace(worldPos, shadowMatrix, reverseZ);

    // if sampleCount is known compileTime, this get rid of the shadowKernel binding
    float shadow;
    if (sampleCount <= 1) {
        shadow = shadow2DProj(shadow_sampler, shadowMap, lightScreen);
    } else {
        //*
        // penumbra
        float filteringSizeFactor = shadowRadius * lightScreen.w;

        //smooth all samples
        float4 samplePos = lightScreen;
        float totalAccum = 0.0;
        for(int i=0; i < sampleCount; i++){
            samplePos.xyz = lightScreen.xyz + shadowKernel[i].xyz * filteringSizeFactor;
            totalAccum += shadow2DProj(shadow_sampler, shadowMap, samplePos);
        }
        /*/ This version could introduce more shadow acne
        float3 uvp = lightScreen.xyz / lightScreen.w;
        float2 texelSize = shadowRadius;

         //smooth all samples
         float totalAccum = 0.0;
         for (int i=0; i < sampleCount; i++){
            float2 samplePos = uvp.xy + texelSize * shadowKernel[i].xy;
            totalAccum  += shadowMap.sample_compare(shadow_sampler, samplePos, uvp.z);
         }
         //*/
        shadow = totalAccum / float(sampleCount);
    }

    // Is this useful ?
    shadow *= step(0., lightScreen.w);

    return shadow;
}

inline float ComputeCascadeBlendAmount(float3 shadowPos, bool cascadeBlending)
{
    const float cascadeBlendingFactor = 0.1f; // No need to configure that

    float3 cascadePos = abs(shadowPos.xyz * 2.f - 1.f);
    
    if (cascadeBlending) {
#if 0
        const float edge = 1.f - cascadeBlendingFactor;
        // could also do a smoothstep
        cascadePos = 1.f - saturate((cascadePos - edge) / cascadeBlendingFactor);
        return cascadePos.x * cascadePos.y * cascadePos.z; //min(o.x, o.y);
#else
        // OPTIM use reduce_max
        float distToEdge = 1.0f - max(max(cascadePos.x, cascadePos.y), cascadePos.z);
        return smoothstep(0.0f, cascadeBlendingFactor, distToEdge);
#endif
    } else {
        return step(cascadePos.x, 1.f) * step(cascadePos.y, 1.f) * step(cascadePos.z, 1.f);
    }
}

inline float4 SampleShadowCascade(sampler shadow_sampler, depth2d_array<float> shadowMaps, float3 shadowPosition, uint cascadeIndex, constant float4* shadowKernel, int sampleCount, float shadowRadius)
{
    // cascade debug + grid
    float2 gridSize = float2(shadowMaps.get_width(), shadowMaps.get_height()) / 32;
    float gd = scn::checkerboard(shadowPosition.xy, gridSize);
    float3 gridCol = mix(float3(scn::debugColorForCascade(cascadeIndex).rgb), float3(0.f), float3(gd > 0.f));
    
    float shadow = 0.f;
    if (sampleCount > 1) {

        // penumbra : sum all samples
        for (int i = 0; i < sampleCount; ++i) {
            shadow += shadow2DArray(shadow_sampler, shadowMaps, shadowKernel[i].xyz * shadowRadius + shadowPosition, cascadeIndex);
        }
        shadow /= float(sampleCount);
    } else {
        // OPTIM : do not use proj version since cascade are never projective
        shadow = shadow2DArray(shadow_sampler, shadowMaps, shadowPosition, cascadeIndex);
    }
    return float4(gridCol, shadow);
}

inline float4 ComputeCascadedShadow(sampler shadow_sampler, float3 viewPos, float4x4 shadowMatrix, constant float4 *cascadeScale, constant float4 *cascadeBias, int cascadeCount, depth2d_array<float> shadowMaps, bool enableCascadeBlending, constant float4* shadowKernel, int sampleCount, float shadowRadius)
{
    float4 shadow = 0.f;
    float opacitySum = 1.f;
    
    // get the position in light space
    float3 pos_ls =  (shadowMatrix * float4(viewPos, 1.f)).xyz;

    for (int c = 0; c < cascadeCount; ++c) {
        
        float3 pos_cs =  pos_ls * cascadeScale[c].xyz + cascadeBias[c].xyz;

        // we multiply the radius by the scale factor of the cascade
        float cascadeRadius = shadowRadius * cascadeScale[c].x;

        float opacity = ComputeCascadeBlendAmount(pos_cs, enableCascadeBlending);
        if (opacity > 0.f) { // this cascade should be considered
            
            float alpha = opacity * opacitySum;
            shadow += SampleShadowCascade(shadow_sampler, shadowMaps, pos_cs, c, shadowKernel, sampleCount, cascadeRadius) * alpha;
            opacitySum -= alpha;
        }
        if (opacitySum <= 0.f) // fully opaque shadow (no more blending needed) -> bail out
            break;
    }

    return shadow;
}




// Macro for layered rendering & shadermodifier

#ifdef USE_LAYERED_RENDERING
#define texture2d_layer texture2d_array
#define sampleLayer(a,b) sample(a,b,in.sliceIndex)
#else
#define texture2d_layer texture2d
#define sampleLayer(a,b) sample(a,b)
#endif

#if defined(HAS_NORMAL) || defined(USE_OPENSUBDIV)
#define HAS_OR_GENERATES_NORMAL 1
#endif


#ifdef C3D_USE_TEXTURE_FOR_LIGHT_INDICES
#define LightIndex(lid) u_lightIndicesTexture.read((ushort)lid).x
#else
#define LightIndex(lid) u_lightIndicesBuffer[lid]
#endif

// Inputs

typedef struct {

#ifdef USE_MODELTRANSFORM
    float4x4 modelTransform;
#endif
#ifdef USE_INVERSEMODELTRANSFORM
    float4x4 inverseModelTransform;
#endif
#ifdef USE_MODELVIEWTRANSFORM
    float4x4 modelViewTransform;
#endif
#ifdef USE_INVERSEMODELVIEWTRANSFORM
    float4x4 inverseModelViewTransform;
#endif
#ifdef USE_NORMALTRANSFORM
    float4x4 normalTransform;
#endif
#ifdef USE_MODELVIEWPROJECTIONTRANSFORM
    float4x4 modelViewProjectionTransform;
#endif
#ifdef USE_INVERSEMODELVIEWPROJECTIONTRANSFORM
    float4x4 inverseModelViewProjectionTransform;
#endif
#ifdef USE_MOTIONBLUR
    float4x4 lastFrameModelTransform;
    float motionBlurIntensity;
#endif
#ifdef USE_BOUNDINGBOX
    float2x3 boundingBox;
#endif
#ifdef USE_WORLDBOUNDINGBOX
    float2x3 worldBoundingBox;
#endif
#ifdef USE_NODE_OPACITY
    float nodeOpacity;
#endif
#if defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 2)
    sh2_coefficients shCoefficients;
#elif defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 3)
    sh3_coefficients shCoefficients;
#endif
#ifdef USE_SKINNING // need to be last since we may cut the buffer size based on the real bone number
    float4 skinningJointMatrices[765]; // Consider having a separate buffer ?
#endif
} commonprofile_node;

typedef struct {
    float3 position         [[attribute(SCNVertexSemanticPosition)]];
#ifdef HAS_NORMAL
    float3 normal           [[attribute(SCNVertexSemanticNormal)]];
#endif
#ifdef USE_TANGENT
    float4 tangent          [[attribute(SCNVertexSemanticTangent)]];
#endif
#ifdef USE_VERTEX_COLOR
    float4 color            [[attribute(SCNVertexSemanticColor)]];
#endif
#ifdef USE_SKINNING
    float4 skinningWeights  [[attribute(SCNVertexSemanticBoneWeights)]];
    uint4  skinningJoints   [[attribute(SCNVertexSemanticBoneIndices)]];
#endif
#ifdef NEED_IN_TEXCOORD0
    float2 texcoord0        [[attribute(SCNVertexSemanticTexcoord0)]];
#endif
#ifdef NEED_IN_TEXCOORD1
    float2 texcoord1        [[attribute(SCNVertexSemanticTexcoord1)]];
#endif
#ifdef NEED_IN_TEXCOORD2
    float2 texcoord2        [[attribute(SCNVertexSemanticTexcoord2)]];
#endif
#ifdef NEED_IN_TEXCOORD3
    float2 texcoord3        [[attribute(SCNVertexSemanticTexcoord3)]];
#endif
#ifdef NEED_IN_TEXCOORD4
    float2 texcoord4        [[attribute(SCNVertexSemanticTexcoord4)]];
#endif
#ifdef NEED_IN_TEXCOORD5
    float2 texcoord5        [[attribute(SCNVertexSemanticTexcoord5)]];
#endif
#ifdef NEED_IN_TEXCOORD6
    float2 texcoord6        [[attribute(SCNVertexSemanticTexcoord6)]];
#endif
#ifdef NEED_IN_TEXCOORD7
    float2 texcoord7        [[attribute(SCNVertexSemanticTexcoord7)]];
#endif
} scn_vertex_t; // __attribute__((scn_per_frame));

typedef struct {
    float4 fragmentPosition [[position]]; // The window relative coordinate (x, y, z, 1/w) values for the fragment
#ifdef USE_POINT_RENDERING
    float fragmentSize [[point_size]];
#endif
#ifdef USE_VERTEX_COLOR
    float4 vertexColor;
#endif
#ifdef USE_PER_VERTEX_LIGHTING
    float3 diffuse;
#ifdef USE_SPECULAR
    float3 specular;
#endif
#endif
#if defined(USE_POSITION) && (USE_POSITION == 2)
    float3 position;
#endif
#if defined(USE_NORMAL) && (USE_NORMAL == 2) && defined(HAS_OR_GENERATES_NORMAL)
    float3 normal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    float3 tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    float3 bitangent;
#endif
#ifdef USE_DISPLACEMENT_MAP
    float2 displacementTexcoord;   // Displacement texture coordinates
#endif
#ifdef USE_NODE_OPACITY
    float nodeOpacity;
#endif
#ifdef USE_TEXCOORD
    
#endif
    
#ifdef USE_EXTRA_VARYINGS
    
#endif
    
#ifdef USE_MOTIONBLUR
    float3 velocity;// [[ center_no_perspective ]];
#endif
#ifdef USE_OUTLINE
    float outlineHash [[ flat ]];
#endif
#if __METAL_VERSION__ >= 200
#ifdef USE_LAYERED_RENDERING
    uint   sliceIndex [[render_target_array_index]];
#endif
#ifdef USE_MULTIPLE_VIEWPORTS_RENDERING
    uint   sliceIndex [[viewport_array_index]];
#endif
#endif
} commonprofile_io;

// Shader modifiers declaration (only enabled if one modifier is present)
#ifdef USE_SHADER_MODIFIERS

#endif

// This may rely on shader modifiers declaration
#define USE_QUAT_FOR_IES 1

// 256 bytes
struct scn_light
{
    float4 color; // color.rgb + shadowColor.a                                      16
    float3 pos; //                                                                  16 (12)
    float3 dir; //                                                                  16 (12)
    float shadowRadius; //                                                          4
    uint8_t lightType; //                                                           1
    uint8_t attenuationType; //                                                     1
    uint8_t shadowSampleCount; //                                                   1
                                //                                                  55  but in reality 64 for alignment reasons
    union {
        struct {
            float4      cascadeScale[4]; // max cascade count
            float4      cascadeBias[4];
        } directional; // 128
        struct {
            float4      attenuationFactors; // scale/bias/power/invSquareRadius                  16
            float3      shadowScaleBias; // xy: scale/bias for z_lin -> z_ndc, z: depth bias
        } omni;
        struct {
            float4      _attenuationFactors; // need to match omni, always use omni.attenuationFactors
            float2      scaleBias; // scale/bias to compute spot falloff
        } spot;
        struct {
            float4      _attenuationFactors; // need to match omni, always use omni.attenuationFactors
            float2      scaleBias; // scale/bias to compute ies LUT
#if USE_QUAT_FOR_IES
            float4      light_from_view_quat; // OPTIM: this could be a simple quaternion
#else
            float4x4    light_from_view; // OPTIM: this could be a simple quaternion
#endif
        } ies;
        struct {
            float3  offset;
            float4  halfExtents; // w: contains the blending distance
            float3  parallaxCenter;
            float3  parallaxExtents;
            int32_t index; // index of the probe in the probe array (do not use uint8_t because of compiler crash)
            int32_t parallaxCorrection; // do not use bool (compiler crash)
        } probe;
    } parameters; // 128

    float4x4    shadowMatrix; // 64

};

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 120

#define ambientOcclusionTexcoord ambientTexcoord

struct SCNShaderSurface {
    float3 view;                // Direction from the point on the surface toward the camera (V)
    float3 position;            // Position of the fragment
    float3 normal;              // Normal of the fragment (N)
    float3 geometryNormal;      // Normal of the fragment - not taking into account normal map
    float2 normalTexcoord;      // Normal texture coordinates
    float3 tangent;             // Tangent of the fragment
    float3 bitangent;           // Bitangent of the fragment
    float4 ambient;             // Ambient property of the fragment
    float2 ambientTexcoord;     // Ambient texture coordinates
    float4 diffuse;             // Diffuse property of the fragment. Alpha contains the opacity.
    float2 diffuseTexcoord;     // Diffuse texture coordinates
    float4 specular;            // Specular property of the fragment
    float2 specularTexcoord;    // Specular texture coordinates
    float4 emission;            // Emission property of the fragment
    float2 emissionTexcoord;    // Emission texture coordinates
    float4 selfIllumination;            // selfIllumination property of the fragment
    float2 selfIlluminationTexcoord;    // selfIllumination texture coordinates
    float4 multiply;            // Multiply property of the fragment
    float2 multiplyTexcoord;    // Multiply texture coordinates
    float4 transparent;         // Transparent property of the fragment
    float2 transparentTexcoord; // Transparent texture coordinates
    float4 reflective;          // Reflective property of the fragment
    float  metalness;           // Metalness
    float2 metalnessTexcoord;   // Metalness texture coordinates
    float  roughness;           // Roughness
    float2 roughnessTexcoord;   // Roughness texture coordinates
    float shininess;            // Shininess property of the fragment.
    float fresnel;              // Fresnel property of the fragment.
    float ambientOcclusion;     // Ambient occlusion term of the fragment
    float3 _normalTS;           // UNDOCUMENTED in tangent space
#ifdef USE_SURFACE_EXTRA_DECL
    
#endif
};

// Structure to gather property of a light, packed to give access in a light shader modifier
// This must be kept intact for back compatibility in lighting modifiers
struct SCNShaderLight {
    float4 intensity;
    float3 direction;

    float  _att;
    float3 _spotDirection;
    float  _distance;
};

enum SCNLightingModel
{
    SCNLightingModelConstant,
    SCNLightingModelLambert,
    SCNLightingModelPhong,
    SCNLightingModelBlinn,
    SCNLightingModelNone,
    SCNLightingModelPhysicallyBased,

    SCNLightingModelCustom // 6 implicit when using a lighting shader modifier
};

enum C3DLightAttenuationType
{
    kC3DLightAttenuationTypeNone,
    kC3DLightAttenuationTypeConstant,
    kC3DLightAttenuationTypeLinear,
    kC3DLightAttenuationTypeQuadratic,
    kC3DLightAttenuationTypeExponent,
    kC3DLightAttenuationTypePhysicallyBased,
};

#define PROBES_NORMALIZATION 0
#define PROBES_OUTER_BLENDING 1

struct SCNShaderLightingContribution
{
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 modulate;

#if PROBES_NORMALIZATION
    float4 probesWeightedSum; // rgb: sum a:normalization factor
#else
    float  probeRadianceRemainingFactor;
#endif

    thread SCNShaderSurface& surface;

#ifdef USE_PER_VERTEX_LIGHTING
    commonprofile_io out;
#else
    commonprofile_io in;
#endif

#if USE_REVERSE_Z
    constant static constexpr bool reverseZ = true;
#else
    constant static constexpr bool reverseZ = false;
#endif

#ifdef USE_PBR
    static constexpr sampler linearSampler = sampler(filter::linear, mip_filter::linear);

    float  selfIlluminationOcclusion;
    float3 reflectance;
    float3 probeReflectance;
    float  NoV;
#endif
    SCNShaderLightingContribution(thread SCNShaderSurface& iSurface, commonprofile_io io):surface(iSurface)
#ifdef USE_PER_VERTEX_LIGHTING
    ,out(io)
#else
    ,in(io)
#endif
    {
        ambient = 0.f;
        diffuse = 0.f;
        specular = 0.f;
#if PROBES_NORMALIZATION
#if PROBES_OUTER_BLENDING
        probesWeightedSum = float4(0.f);
#else
        probesWeightedSum = float4(0.f, 0.f, 0.f, 0.000001f); // avoid divide by 0 with an epsilon
#endif
#else
        probeRadianceRemainingFactor = 1.f;
#endif

#ifdef USE_MODULATE
        modulate = 1.f;
#else
        modulate = 0.f;
#endif
    }

#ifdef USE_PBR
    void prepareForPBR(texture2d<float, access::sample> specularDFG, float occ)
    {
        selfIlluminationOcclusion = occ;

        float3 n = surface.normal;
        float3 v = surface.view;
        reflectance = mix(float3(PBR_F0_NON_METALLIC), surface.diffuse.rgb, surface.metalness);

        NoV = saturate(dot(n, v));
        float2 DFG = specularDFG.sample(linearSampler, float2(NoV, surface.roughness)).rg;
        probeReflectance = reflectance * DFG.r + DFG.g;
    }
#endif


#ifdef USE_LIGHT_MODIFIER
    
#endif

    float4 debug_pixel(float2 fragmentPosition)
    {
        const int width = 64;
        switch (int(fragmentPosition.x + fragmentPosition.y ) / width) {
            case 0: return float4(surface.view, 1.f);
            case 1: return float4(surface.position, 1.f);
            case 2: return float4(surface.normal, 1.f);
            case 3: return float4(surface.geometryNormal, 1.f);
            case 4: return float4(float3(surface.ambientOcclusion), 1.f);
            case 5: return surface.diffuse;
            case 6: return float4(float3(surface.metalness), 1.f);
            case 7: return float4(float3(surface.roughness), 1.f);
            case 8: return float4(ambient, 1.f);
            case 9: return float4(diffuse, 1.f);
            default: return float4(specular, 1.f);
        }
    }

    // tool functions, could be external

    static inline float3 lambert_diffuse(float3 l, float3 n, float3 color, float intensity) {
        return color * (intensity * saturate(dot(n, l)));
    }

    void lambert(float3 l, float3 color, float intensity)
    {
        diffuse += lambert_diffuse(l, surface.normal, color, intensity);
    }

    void blinn(float3 l, float3 color, float intensity)
    {
        float3 D = lambert_diffuse(l, surface.normal, color, intensity);
        diffuse += D;

        float3 h = normalize(l + surface.view);
        specular += powr(saturate(dot(surface.normal, h)), surface.shininess) * D;
    }

    void phong(float3 l, float3 color, float intensity)
    {
        float3 D = lambert_diffuse(l, surface.normal, color, intensity);
        diffuse += D;

        float3 r = reflect(-l, surface.normal);
        specular += powr(saturate(dot(r, surface.view)), surface.shininess) * D;
    }

#ifdef USE_PBR
    void pbr(float3 l, float3 color, float intensity)
    {
        float3 n = surface.normal;
        float3 v = surface.view;

        float3 h = normalize(l + v);

        float NoL = saturate(dot(n, l));
        float NoH = saturate(dot(n, h));
        float LoH = saturate(dot(l, h));

        float roughness = surface.roughness;
        float alpha = roughness * roughness; // perceptually-linear roughness

        float D   = scn_brdf_D(alpha, NoH);
        float3 F  = scn_brdf_F_opt(reflectance, LoH);
        float Vis = scn_brdf_V(alpha, NoL, NoV);

        // keep the scalar separated
        diffuse  += color * (NoL * M_1_PI_F * intensity);
        specular += color * F * ( NoL * D * Vis * intensity);
    }
#endif

    void custom(float3 _l, float3 _color, float _intensity)
    {
#ifdef USE_LIGHT_MODIFIER
        thread SCNShaderLightingContribution &_lightingContribution = *this;
        thread SCNShaderSurface& _surface = surface;
        SCNShaderLight _light = {.direction = _l, .intensity = float4(_color, 1.f), ._att = _intensity };
        // DoLightModifier START
        
        // DoLightModifier END
#endif
    }

    void shade(float3 l, float3 color, float intensity)
    {
#ifdef LIGHTING_MODEL
        switch (LIGHTING_MODEL) {
            case SCNLightingModelLambert:           lambert(l, color, intensity); break;
            case SCNLightingModelBlinn:             blinn(l, color, intensity);   break;
            case SCNLightingModelPhong:             phong(l, color, intensity);   break;
#ifdef USE_PBR
            case SCNLightingModelPhysicallyBased:   pbr(l, color, intensity);     break;
#endif
            case SCNLightingModelCustom:            custom(l, color, intensity);  break;
            default:                                                                       break; // static_assert(0, "should not go there");
        }
#endif
    }

    // this implementation seems more correct as it never goes larger than when near (dist<1)
    // https://imdoingitwrong.wordpress.com/2011/01/31/light-attenuation/
    // cutoff must be computed with Eq(1) with lightAttenuationEnd and stored in the light struct
    float pbr_dist_attenuation_alternate(float3 l, float cutoff) {
        //        float radius = 0.01f; // consider point lights as 1cm
        float radius = 0.1f; // consider point lights as 10cm
        float factor = 1.f / (1.f + length(l)/radius);
        float attenuation = saturate(factor * factor); // Eq(1)
        return saturate((attenuation - cutoff) / (1.f - cutoff));
    }

    float pbr_dist_attenuation(float3 l, float inv_square_radius) {
        float sqr_dist = length_squared(l);
        float atten = 1.f / max(sqr_dist, 0.0001f);

        // smoothing factor to avoid hard clip of the lighting
        float factor = saturate(1.f - scn::sq(sqr_dist * inv_square_radius));
        return atten * factor * factor;
    }

    float dist_attenuation(float3 l, float4 att, C3DLightAttenuationType type)
    {
        switch (type) {
            case kC3DLightAttenuationTypeConstant:
                return step(length(l), att.x);
            case kC3DLightAttenuationTypeLinear:
                return saturate(length(l) * att.x + att.y);
            case kC3DLightAttenuationTypeQuadratic:
                return scn::sq(saturate(length(l) * att.x + att.y));
            case kC3DLightAttenuationTypeExponent:
                return powr(saturate(length(l) * att.x + att.y), att.z);
            case kC3DLightAttenuationTypePhysicallyBased:
                return pbr_dist_attenuation(l, att.w);
            default: // case kC3DLightAttenuationTypeNone:
                return 1.f;
        }
    }

    float spot_attenuation(float3 l, scn_light light)
    {
        // only support linear attenuation (spotExponent is SPI)
        return saturate(dot(l, light.dir) * light.parameters.spot.scaleBias.x + light.parameters.spot.scaleBias.y);
    }

    void shade_modulate(float3 l, float4 color, float intensity)
    {
        constexpr half3 white = half3(1.h);
        // color.a contains the gobo slot intensity -> used to fade it
        modulate *= float3(mix(white, half3(color.rgb), half(color.a * intensity)));
    }

    float3 gobo(float3 pos, scn_light light, texture2d<half> goboTexture, sampler goboSampler)
    {
        half3 g = texture2DProj(goboTexture, goboSampler, (light.shadowMatrix * float4(pos, 1.f))).rgb;
        return light.color.rgb * float3(mix(1.h, g, half(light.color.a)));
    }

    float shadow(float3 pos, scn_light light, depth2d<float> shadowMap)
    {
        float shadow = ComputeShadow(scn_shadow_sampler, pos, light.shadowMatrix, shadowMap, reverseZ);
        return 1.f - shadow * light.color.a; // shadow color
    }

    // this versions takes the sample count from the light. This forces dynamic loops, not a good idea on iOS
    float shadow(float3 pos, scn_light light, depth2d<float> shadowMap, constant float4* shadowKernel)
    {
        float shadow = ComputeSoftShadow(scn_shadow_sampler, pos, light.shadowMatrix, shadowMap, shadowKernel, light.shadowSampleCount, light.shadowRadius, reverseZ);
        return 1.f - shadow * light.color.a; // shadow color
    }

    float shadow(float3 pos, scn_light light, depth2d<float> shadowMap, constant float4* shadowKernel, int shadowSampleCount)
    {
        float shadow = ComputeSoftShadow(scn_shadow_sampler, pos, light.shadowMatrix, shadowMap, shadowKernel, shadowSampleCount, light.shadowRadius, reverseZ);
        return 1.f - shadow * light.color.a; // shadow color
    }

    float shadow(float3 pos, scn_light light, depth2d<float> shadowMap, int shadowSampleCount)
    {
        float shadow = ComputeSoftShadowGrid(scn_shadow_sampler, pos, light.shadowMatrix, shadowMap, shadowSampleCount, reverseZ);
        return 1.f - shadow * light.color.a; // shadow color
    }

    float shadow_omni(float3 pos_vs, float3 nrm_vs, scn_light light, depthcube<float> shadowMap, constant float4* shadowKernel, int sampleCount)
    {
        // else use hemispheric sampling
#define USE_TANGENT_SAMPLING 0

        float2 scaleBias = light.parameters.omni.shadowScaleBias.xy;
        float  depthBias = light.parameters.omni.shadowScaleBias.z;

        // offset/bias for shadow acne
        pos_vs += nrm_vs * depthBias;

        // transform pos from view space to light space
        float3 pos_ls = (light.shadowMatrix * float4(pos_vs, 1.f)).xyz;

        // compute z_lin for sample cube face (symetric so +x == -x, etc)
        float z_lin = scn::reduce_max(abs(pos_ls));

        // if we want to clip the shadows to the far planes of the cube...
        // if (z_lin > zFar)
        //   return 1.f;

        // transform linear_z to depthbuffer_z
        float z_ndc = (z_lin * scaleBias.x + scaleBias.y) / z_lin;

        // if sampleCount is known compileTime, this get rid of the shadowKernel binding
        float shadow;
        if (sampleCount <= 1) {
            shadow = shadowMap.sample_compare(scn_shadow_sampler, pos_ls.xyz, z_ndc);
        } else {
            // penumbra
            float filteringSizeFactor = light.shadowRadius;

#if USE_TANGENT_SAMPLING
            float3 tgt_x, tgt_y;
            scn::orthogonal_basis(pos_ls, tgt_x, tgt_y);
#else
            float3 nrm_ls = (light.shadowMatrix * float4(nrm_vs, 0.f)).xyz;
#endif

            //smooth all samples
            float totalAccum = 0.0;
            for(int i=0; i < sampleCount; i++){

#if USE_TANGENT_SAMPLING
                float2 scale = shadowKernel[i].xy * filteringSizeFactor * 2.f;
                float3 smp_ls = pos_ls.xyz + tgt_x * scale.x + tgt_y * scale.y;
#else
                float3 smp_ls = pos_ls.xyz + scn::randomHemisphereDir(nrm_ls, shadowKernel[i].xy) * filteringSizeFactor;
#endif

                // Do we want to compare with reference depth or smp depth?
                // z_lin = scn::reduce_max(abs(smp_ls));
                // z_ndc = (z_lin * scaleBias.x + scaleBias.y) / z_lin;

                totalAccum += shadowMap.sample_compare(scn_shadow_sampler, smp_ls, z_ndc);
            }
            shadow = totalAccum / float(sampleCount);
        }

        return 1.f - shadow * light.color.a; // shadow color
    }

    float shadow(float3 pos, constant scn_light& light, depth2d_array<float> shadowMaps, int cascadeCount, bool blendCascade, constant float4* shadowKernel, int sampleCount)
    {
        float shadow = ComputeCascadedShadow(scn_shadow_sampler, pos, light.shadowMatrix, light.parameters.directional.cascadeScale, light.parameters.directional.cascadeBias, cascadeCount, shadowMaps, blendCascade, shadowKernel, sampleCount, light.shadowRadius).a;
        return 1.f - shadow * light.color.a; // shadow color
    }

    // MARK: Directional

    void add_directional(scn_light light)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        shade(light.dir, light.color.rgb, intensity);
    }

    // gobo
    void add_directional(scn_light light, texture2d<half> goboTexture, sampler goboSampler, bool modulated)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        light.color.rgb = gobo(surface.position, light, goboTexture, goboSampler);
        if (modulated) {
            shade_modulate(light.dir, light.color, 1.f);
        } else {
            shade(light.dir, light.color.rgb, intensity);
        }
    }

    // support simple shadows (no soft, no cascade)
    void add_directional(scn_light light, depth2d<float> shadowMap)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        intensity *= shadow(surface.position, light, shadowMap);
        shade(light.dir, light.color.rgb, intensity);
    }

    // support soft shadows (non cascaded, dynamic sample count)
    void add_directional(scn_light light, depth2d<float> shadowMap, constant float4* shadowKernel)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        intensity *= shadow(surface.position, light, shadowMap, shadowKernel);
        shade(light.dir, light.color.rgb, intensity);
    }

    void add_directional(scn_light light, depth2d<float> shadowMap, constant float4* shadowKernel, int sampleCount)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        intensity *= shadow(surface.position, light, shadowMap, shadowKernel, sampleCount);
        shade(light.dir, light.color.rgb, intensity);
    }

    // regular grid PCF
    void add_directional(scn_light light, depth2d<float> shadowMap, int sampleCount)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        intensity *= shadow(surface.position, light, shadowMap, sampleCount);
        shade(light.dir, light.color.rgb, intensity);
    }

    // version supporting cascade shadows
    void add_directional(constant scn_light& light, depth2d_array<float> shadowMaps, int cascadeCount, bool blendCascade, constant float4* shadowKernel, int sampleCount, bool debugCascades)
    {
#ifdef USE_PBR
        float intensity = M_PI_F;
#else
        float intensity = 1.f;
#endif
        if (debugCascades) {
            float4 shadowDebug = ComputeCascadedShadow(scn_shadow_sampler, surface.position, light.shadowMatrix, light.parameters.directional.cascadeScale, light.parameters.directional.cascadeBias, cascadeCount, shadowMaps, blendCascade, shadowKernel, sampleCount, light.shadowRadius);
            intensity *= (1.f - shadowDebug.a);
            shade(light.dir, light.color.rgb, intensity);
            diffuse.rgb = mix(diffuse.rgb, shadowDebug.rgb, light.color.a);
        } else {
            intensity *= shadow(surface.position, light, shadowMaps, cascadeCount, blendCascade, shadowKernel, sampleCount);
            shade(light.dir, light.color.rgb, intensity);
        }
    }

    // MARK: Omni

    float dist_attenuation(float3 unnormalized_l, scn_light light, bool local = false)
    {
#ifdef USE_PBR
        return 1000.f * pbr_dist_attenuation(unnormalized_l, light.parameters.omni.attenuationFactors.w);
        // This alternate model seems to better fit V-Ray render.... To confirm
        // float intensity = 1000.f * pbr_dist_attenuation_alternate(unnormalized_l, 0.f);
#else
        if (local)
            return dist_attenuation(unnormalized_l, light.parameters.omni.attenuationFactors, kC3DLightAttenuationTypeExponent);
        else
            return 1.f;
#endif
    }
    
    void add_omni(scn_light light)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        shade(l, light.color.rgb, dist_attenuation(unnormalized_l, light));
    }

    void add_omni(scn_light light, depthcube<float> shadowMap, constant float4* shadowKernel, int sampleCount)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity *= shadow_omni(surface.position, surface.normal, light, shadowMap, shadowKernel, sampleCount);
        shade(l, light.color.rgb, intensity);
    }

    void add_local_omni(scn_light light)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        shade(l, light.color.rgb, dist_attenuation(unnormalized_l, light, true));
    }

    // MARK: Spot

    void add_spot(scn_light light)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity      *= spot_attenuation(l, light);
        shade(l, light.color.rgb, intensity);
    }

    void add_spot(scn_light light, texture2d<half> goboTexture, sampler goboSampler, bool modulated)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity      *= spot_attenuation(l, light);
        light.color.rgb = gobo(surface.position, light, goboTexture, goboSampler);
        if (modulated) {
            shade_modulate(l, light.color, intensity);
        } else {
            shade(l, light.color.rgb, intensity);
        }
    }

    void add_local_spot(scn_light light)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light, true);
        intensity      *= spot_attenuation(l, light);
        shade(l, light.color.rgb, intensity);
    }

    // support simple shadows (non cascaded)
    void add_spot(scn_light light, depth2d<float> shadowMap, constant float4* shadowKernel, int sampleCount)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity      *= spot_attenuation(l, light);
        intensity      *= shadow(surface.position, light, shadowMap, shadowKernel, sampleCount);
        shade(l, light.color.rgb, intensity);
    }

    // MARK: Probe

#ifdef USE_PBR

    // MARK: Radiance

#ifdef C3D_SUPPORT_CUBE_ARRAY
    void add_local_probe(scn_light light, texturecube_array<half> probeTextureArray)
#else
    void add_local_probe(scn_light light, texture2d_array<half> probeTextureArray)
#endif
    {
#if !PROBES_NORMALIZATION
        if (probeRadianceRemainingFactor <= 0.f)
            return;
#endif

        bool parallaxCorrection = light.parameters.probe.parallaxCorrection;
        int    probeIndex       = light.parameters.probe.index;
        float3 probeExtents     = light.parameters.probe.halfExtents.xyz;
        float  blendDist        = light.parameters.probe.halfExtents.w;
        float3 probeOffset      = light.parameters.probe.offset;
        float3 parallaxExtents  = light.parameters.probe.parallaxExtents;
        float3 parallaxCenter   = light.parameters.probe.parallaxCenter;

        float3 n = surface.normal;
        float3 v = surface.view;
        float3 r = reflect(-v, n); // mirror vector (view vector around normal)

        float3 specDir = scn::mat4_mult_float3(light.shadowMatrix, r);

        // TODO blend weight ? accumulate in alpha and normalize ?
        float3 pos_ls = (light.shadowMatrix * float4(surface.position, 1.f)).xyz;

        // OPTIM: we should be able to multiply by the extents in CPU in matrix and use 1 here...
        float3 d = abs(pos_ls) - probeExtents;
#if PROBES_OUTER_BLENDING
        if (any(d > blendDist))
#else
        if (any(d > 0.f))
#endif
        {
            return;
        }

#if PROBES_NORMALIZATION
        //      inside    '  |  '   outside
        // nd      1      1 0.5 0      0      (per component)
#if PROBES_OUTER_BLENDING
        float3 nd = saturate(-(d / blendDist) * 0.5f + 0.5f);
#else
        float3 nd = saturate(-(d / blendDist));
#endif
        float probeFactor = (nd.x * nd.y * nd.z) * light.color.r;
#else
        // signed distance in the probe box
        float sd = min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
#if PROBES_OUTER_BLENDING
        float probeFactor = saturate(1.f - sd / blendDist);
#else
        float probeFactor = saturate(-sd / blendDist);
#endif
        //      inside      |  '   outside
        // sd     -x        0  b      +x
        // fc      1        1  0       0
        probeFactor *= probeRadianceRemainingFactor * light.color.r; // Do we really need this or 1.f is enough (we need this for global probe anyway...)
#endif

        if (parallaxCorrection /* && all(d < 0.f) */) {
            // OPTIM: we should be able to multiply by the extents in CPU and use 1 here...
            float3 pos_off = pos_ls + parallaxCenter;
            float3 t1 = ( parallaxExtents - pos_off) / specDir;
            float3 t2 = (-parallaxExtents - pos_off) / specDir;
            float3 tmax = max(max(0, t1), t2); // max(0, ..) to avoid correction when outside the box (in the blending zone)
            float t = min(tmax.x, min(tmax.y, tmax.z));

            // Use Distance in WS directly to recover intersection
            float3 hit_ls = pos_ls + specDir * t;
            specDir = hit_ls - probeOffset;
        }

        float mipd = float(probeTextureArray.get_num_mip_levels()) - 1.f;
        const float intensity = surface.ambientOcclusion * probeFactor;

        float mips = surface.roughness * mipd;
#ifdef C3D_SUPPORT_CUBE_ARRAY
        float3 LD = float3(probeTextureArray.sample(linearSampler, specDir, probeIndex, level(mips)).rgb);
#else
        float2 specUV = scn::dual_paraboloid_from_cartesian(normalize(specDir));
        float3 LD = float3(probeTextureArray.sample(linearSampler, specUV, probeIndex, level(mips)).rgb);
#endif

        /* Debug blending with primary colors
        switch (probeIndex) {
            case 1: LD = float3(1.f, 0.f, 0.f); break;
            case 2: LD = float3(0.f, 1.f, 0.f); break;
            case 3: LD = float3(0.f, 0.f, 1.f); break;
            default: LD = float3(1.f, 1.f, 1.f); break;
        }*/

        // radiance
#if PROBES_NORMALIZATION
        probesWeightedSum += float4(LD * intensity * probeReflectance, probeFactor);
#else
        probeRadianceRemainingFactor = saturate(probeRadianceRemainingFactor - probeFactor);
        specular += LD * intensity * probeReflectance;
#endif
    }

    void add_global_probe(float4x4 localDirToWorldCubemapDir, float environmentIntensity,
#ifdef C3D_SUPPORT_CUBE_ARRAY
                          texturecube_array<half> probeTextureArray
#else
                          texture2d_array<half> probeTextureArray
#endif
                          )
    {
        float3 n = surface.normal;
        float3 v = surface.view;
        float3 r = reflect(-v, n); // mirror vector (view vector around normal)

        float3 specDir = scn::mat4_mult_float3(localDirToWorldCubemapDir, r);

        float mipd = float(probeTextureArray.get_num_mip_levels()) - 1.f;
        const float intensity = surface.ambientOcclusion * environmentIntensity;

        float mips = surface.roughness * mipd;
#ifdef C3D_SUPPORT_CUBE_ARRAY
        float3 LD = float3(probeTextureArray.sample(linearSampler, specDir, 0, level(mips)).rgb);
#else
        float2 specUV = scn::dual_paraboloid_from_cartesian(normalize(specDir));
        float3 LD = float3(probeTextureArray.sample(linearSampler, specUV, 0, level(mips)).rgb);
#endif

        // radiance
        specular += LD * intensity * probeReflectance;
    }

    void add_global_probe(texturecube<float, access::sample> specularLD,
                          float4x4                           localDirToWorldCubemapDir,
                          float                              environmentIntensity)
    {
        float3 n        = surface.normal;
        float3 v        = surface.view;
        float3 r        = reflect(-v, n); // mirror vector (view vector around normal)
        float roughness = surface.roughness;
        float roughness2= roughness * roughness;

#if 1
        float smoothness = 1.0f - roughness2;
        float specularLerpFactor = (1. - smoothness * (sqrt(smoothness) + roughness2));

        // This does have an effect on smooth object : seems buggy (plus costly)
        //float NoV       = 1.f - saturate(dot(n, v));
        //specularLerpFactor = saturate(specularLerpFactor + 2.f*NoV*NoV*NoV);

        float3 specularDominantNDirection = mix(r, n, specularLerpFactor); // no need to normalize as we fetch in a cubemap
#else
        float3 specularDominantNDirection = r;
#endif
        

        // Specular
        float mipLevel = roughness * float(specularLD.get_num_mip_levels() - 1);

#if 0 // Seamless cubemap filtering
        float3 dirAbs = abs(specularDominantNDirection);
        float dirNormInf = max(dirAbs.x, max(dirAbs.y, dirAbs.z));
        float scale = 1.0f - exp2(mipLevel) / float(specularLD.get_width());
        if (dirAbs.x != dirNormInf) specularDominantNDirection.x *= scale;
        if (dirAbs.y != dirNormInf) specularDominantNDirection.y *= scale;
        if (dirAbs.z != dirNormInf) specularDominantNDirection.z *= scale;
#endif

        float3 LD = specularLD.sample(linearSampler, scn::mat4_mult_float3(localDirToWorldCubemapDir, specularDominantNDirection), level(mipLevel)).rgb;

#if 0 // Specular occlusion - not physically correct
        float specularOcclusion = saturate(pow(NoV + surface.ambientOcclusion, exp2(-16.0f * roughness - 1.0f)) - 1.0f + surface.ambientOcclusion);
        LD *= specularOcclusion;
#endif

        // effectiveAlbedo is multiplied in combine
        specular += LD * (surface.ambientOcclusion * environmentIntensity) * probeReflectance;
    }

    // MARK: Irradiance

    void add_irradiance_from_selfIllum()
    {
        float selfIlluminationAO = saturate(mix(1.f, surface.ambientOcclusion, selfIlluminationOcclusion));
        float3 irradiance = selfIlluminationAO * surface.selfIllumination.rgb;
        diffuse  += irradiance;
    }

    void add_global_irradiance_from_sh(float4x4         localDirToWorldCubemapDir,
#if defined(USE_PROBES_LIGHTING) && (USE_PROBES_LIGHTING == 2)
                                       sh2_coefficients shCoefficients)
#else
    sh3_coefficients shCoefficients)
#endif
    {
        float3 n_sh_space = scn::mat4_mult_float3(localDirToWorldCubemapDir, surface.normal);
        float3 irradiance = shEvalDirection(float4(n_sh_space, 1.), shCoefficients);
        diffuse  += surface.ambientOcclusion * irradiance;
    }

    void add_global_irradiance_probe(texturecube<float, access::sample> irradianceTexture,
                                     float4x4                           localDirToWorldCubemapDir,
                                     float                              environmentIntensity)
    {
        float3 n_cube_space = scn::mat4_mult_float3(localDirToWorldCubemapDir, surface.normal);
        float3 irradiance = irradianceTexture.sample(linearSampler, n_cube_space).rgb;
        diffuse  += (surface.ambientOcclusion * environmentIntensity) * irradiance;
    }

#endif // USE_PBR

    // MARK: IES

    static constexpr sampler iesSampler = sampler(filter::linear, mip_filter::none, address::clamp_to_edge);



    float ies_attenuation(float3 l, scn_light light, texture2d<half> iesTexture)
    {
#if USE_QUAT_FOR_IES
        float3 v    = scn::quaternion_rotate_vector(light.parameters.ies.light_from_view_quat, -l);
#else
        float3 v    = scn::matrix_rotate(light.parameters.ies.light_from_view, -l);
#endif
        float phi   = (v.z * light.parameters.ies.scaleBias.x + light.parameters.ies.scaleBias.y);
        float theta = atan2(v.y, v.x) * 0.5f * M_1_PI_F;
        return iesTexture.sample(iesSampler, float2(phi, abs(theta))).r;
    }

    void add_ies(scn_light light, texture2d<half> iesTexture)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity      *= ies_attenuation(l, light, iesTexture);
        shade(l, light.color.rgb, intensity);
    }

    void add_ies(scn_light light, texture2d<half> iesTexture, depth2d<float> shadowMap, constant float4* shadowKernel, int sampleCount)
    {
        float3 unnormalized_l = light.pos - surface.position;
        float3 l = normalize(unnormalized_l);
        float intensity = dist_attenuation(unnormalized_l, light);
        intensity      *= ies_attenuation(l, light, iesTexture);
        intensity      *= shadow(surface.position, light, shadowMap, shadowKernel, sampleCount);
        shade(l, light.color.rgb, intensity);
    }
};

#endif // #if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 120



enum C3DColorMask {
    kC3DColorMaskRed    = 0x1 << 3,
    kC3DColorMaskGreen  = 0x1 << 2,
    kC3DColorMaskBlue   = 0x1 << 1,
    kC3DColorMaskAlpha  = 0x1 << 0
};

float4 colorFromMask(float4 col, int mask)
{
    switch (mask) {

        case kC3DColorMaskRed:                      return col.r;
        case kC3DColorMaskRed|kC3DColorMaskGreen:   return float4(col.rg, 0.f, 1.f);
        case kC3DColorMaskRed|kC3DColorMaskBlue:    return float4(col.rb, 0.f, 1.f);
        case kC3DColorMaskRed|kC3DColorMaskAlpha:   return float4(col.ra, 0.f, 1.f);

        case kC3DColorMaskGreen:                    return col.g;
        case kC3DColorMaskGreen|kC3DColorMaskBlue:  return float4(col.bg, 0.f, 1.f);
        case kC3DColorMaskGreen|kC3DColorMaskAlpha: return float4(col.ag, 0.f, 1.f);

        case kC3DColorMaskBlue:     return col.b;
        case kC3DColorMaskBlue|kC3DColorMaskAlpha:  return float4(col.ab, 0.f, 1.f);

        case kC3DColorMaskAlpha:    return col.a;
    }
    return col;
}

#ifndef USE_PBR

inline float3 illuminate(SCNShaderSurface surface, SCNShaderLightingContribution lighting)
{
    float3 albedo = surface.diffuse.rgb * surface.ambientOcclusion;
    float3 color = lighting.diffuse * albedo;
#if defined(USE_AMBIENT_LIGHTING) && (defined(LOCK_AMBIENT_WITH_DIFFUSE) || defined(USE_AMBIENT_AS_AMBIENTOCCLUSION))
    color +=  lighting.ambient * albedo;
#endif
#ifdef USE_SELFILLUMINATION
    color += surface.diffuse.rgb * surface.selfIllumination.rgb;
#endif
    
    // Do we want to clamp there ????
    
#ifdef USE_SPECULAR
    float3 S = lighting.specular;
#elif defined(USE_REFLECTIVE)
    float3 S = float3(0.);
#endif
#ifdef USE_REFLECTIVE
    S += surface.reflective.rgb * surface.ambientOcclusion;
#endif
#ifdef USE_SPECULAR
    S *= surface.specular.rgb;
#endif
#if (defined(USE_SPECULAR) || defined(USE_REFLECTIVE)) && !defined(DISABLE_SPECULAR)
    color += S;
#endif
#if defined(USE_AMBIENT) && !defined(USE_AMBIENT_AS_AMBIENTOCCLUSION)
    color += surface.ambient.rgb * lighting.ambient;
#endif
#ifdef USE_EMISSION
    color += surface.emission.rgb;
#endif
#ifdef USE_MULTIPLY
    color *= surface.multiply.rgb;
#endif
#ifdef USE_MODULATE
    color *= lighting.modulate;
#endif
    return color;
}
#endif

struct SCNShaderGeometry
{
    float4 position;
    float3 normal;
    float4 tangent;
    float4 color;
    float pointSize;
    float2 texcoords[8]; // MAX_UV
};

struct commonprofile_uniforms {
    // [id(0)]]
    float4 diffuseColor;
    float4 specularColor;
    float4 ambientColor;
    float4 emissionColor;
    float4 selfIlluminationColor;
    float4 reflectiveColor;
    float4 multiplyColor;
    float4 transparentColor;
    float metalness;
    float roughness;
    // [id(10)]]
    float diffuseIntensity;
    float specularIntensity;
    float normalIntensity;
    float ambientIntensity;
    float emissionIntensity;
    float selfIlluminationIntensity;
    float reflectiveIntensity;
    float multiplyIntensity;
    float transparentIntensity;
    float metalnessIntensity;
    // [id(20)]]
    float roughnessIntensity;
    float displacementIntensity;
    float materialShininess;
    float selfIlluminationOcclusion;
    float transparency;
    float3 fresnel; // x: ((n1-n2)/(n1+n2))^2 y:1-x z:exponent
#if USE_ARGUMENT_BUFFERS
    //[[id(26)]]
    texture2d<float>    emissionTexture;
    sampler             emissionSampler;
    texture2d<float>    ambientTexture;
    sampler             ambientSampler;
    //[[id(30)]]
    texture2d<float>    diffuseTexture;
    sampler             diffuseSampler;
    texture2d<float>    specularTexture;
    sampler             specularSampler;
#if defined(USE_REFLECTIVE_CUBEMAP)
    texturecube<float>  reflectiveTexture;
#else
    texture2d<float>    reflectiveTexture;
#endif
    sampler             reflectiveSampler;
    texture2d<float>    transparentTexture;
    sampler             transparentSampler;
    texture2d<float>    multiplyTexture;
    sampler             multiplySampler;
    //[[id(41)]]
    texture2d<float>    normalTexture;
    sampler             normalSampler;
    texture2d<float>    selfIlluminationTexture;
    sampler             selfIlluminationSampler;
    texture2d<float>    metalnessTexture;
    sampler             metalnessSampler;
    texture2d<float>    roughnessTexture;
    sampler             roughnessSampler;
    texture2d<float>    displacementTexture;
    sampler             displacementSampler;
    //[[id(51)]]
    
#endif // USE_ARGUMENT_BUFFERS
#ifdef TEXTURE_TRANSFORM_COUNT
    float4x4 textureTransforms[TEXTURE_TRANSFORM_COUNT];
#endif
};

#ifdef USE_OPENSUBDIV



struct osd_packed_vertex {
    packed_float3 position;
#if defined(OSD_USER_VARYING_DECLARE_PACKED)
    OSD_USER_VARYING_DECLARE_PACKED
#endif
};

#endif


#ifdef USE_DISPLACEMENT_MAP
static void applyDisplacement(texture2d<float>                 displacementTexture,
                              sampler                          displacementTextureSampler,
                              float2                           displacementTexcoord,
                              thread SCNShaderGeometry&        geometry,
                              constant commonprofile_uniforms& scn_commonprofile)
{
#ifdef USE_DISPLACEMENT_TEXTURE_COMPONENT
	float altitude = colorFromMask(displacementTexture.sample(displacementTextureSampler, displacementTexcoord), USE_DISPLACEMENT_TEXTURE_COMPONENT).r;
#ifdef USE_DISPLACEMENT_INTENSITY
	altitude *= scn_commonprofile.displacementIntensity;
#endif
#if defined(USE_NORMAL) && defined(HAS_OR_GENERATES_NORMAL)
	float3 bitangent = geometry.tangent.w * normalize(cross(geometry.tangent.xyz, geometry.normal.xyz));
	geometry.position.xyz += geometry.normal * altitude;
	
	float3 offset = float3(1.f / displacementTexture.get_width(), 1.f / displacementTexture.get_height(), 0.f);
	float3 h;
	h.x = colorFromMask(displacementTexture.sample(displacementTextureSampler, displacementTexcoord), USE_DISPLACEMENT_TEXTURE_COMPONENT).r;
	h.y = colorFromMask(displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.xz), USE_DISPLACEMENT_TEXTURE_COMPONENT).r;
	h.z = colorFromMask(displacementTexture.sample(displacementTextureSampler, displacementTexcoord-offset.zy), USE_DISPLACEMENT_TEXTURE_COMPONENT).r;
	
#ifdef USE_DISPLACEMENT_INTENSITY
	h *= scn_commonprofile.displacementIntensity;
#endif
	
	float3 n = normalize( float3( (h.x - h.y)/offset.x, 1., (h.x - h.z)/offset.y) );
	geometry.normal = geometry.tangent.xyz * n.x + geometry.normal.xyz * n.y + bitangent.xyz * n.z;
	geometry.tangent.xyz = normalize(cross(bitangent, geometry.normal));
#endif // USE_NORMAL
#else // USE_DISPLACEMENT_TEXTURE_COMPONENT
	float3 displacement = displacementTexture.sample(displacementTextureSampler, displacementTexcoord).rgb;
#ifdef USE_DISPLACEMENT_INTENSITY
	displacement *= scn_commonprofile.displacementIntensity;
#endif
#if defined(USE_NORMAL) && defined(HAS_OR_GENERATES_NORMAL)
	float3 bitangent = geometry.tangent.w * normalize(cross(geometry.tangent.xyz, geometry.normal.xyz));
	geometry.position.xyz += geometry.tangent.xyz * displacement.x + geometry.normal.xyz * displacement.y + bitangent.xyz * displacement.z;
	
	float3 offset = float3(1.f / displacementTexture.get_width(), 1.f / displacementTexture.get_height(), 0.f);
	float3 a = displacementTexture.sample(displacementTextureSampler, displacementTexcoord).rgb;
	float3 b = displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.xz).rgb;
	float3 c = displacementTexture.sample(displacementTextureSampler, displacementTexcoord+offset.zy).rgb;
	
#ifdef USE_DISPLACEMENT_INTENSITY
	a *= scn_commonprofile.displacementIntensity;
	b *= scn_commonprofile.displacementIntensity;
	c *= scn_commonprofile.displacementIntensity;
#endif
	
	b += offset.xzz;
	c -= offset.zzy;
	float3 n = (normalize( cross( b-a, c-a ) ));
	geometry.normal = geometry.tangent.xyz * n.x + geometry.normal.xyz * n.y + bitangent.xyz * n.z;
	geometry.tangent.xyz = normalize(cross(bitangent, geometry.normal));
#endif // USE_NORMAL
#endif // USE_DISPLACEMENT_TEXTURE_COMPONENT
}
#endif // USE_DISPLACEMENT_MAP

#ifdef USE_OUTLINE
static inline float hash(float2 p)
{
    const float2 kMod2 = float2(443.8975f, 397.2973f);
    p  = fract(p * kMod2);
    p += dot(p.xy, p.yx+19.19f);
    return fract(p.x * p.y);
}
#endif

//
// MARK: - Vertex and post-tessellation vertex functions
//

#if defined(USE_TESSELLATION)
struct scn_patch_t {
    patch_control_point<scn_vertex_t> controlPoints;
};
#endif

#if defined(USE_OPENSUBDIV)
#if OSD_IS_ADAPTIVE
[[ patch(quad, VERTEX_CONTROL_POINTS_PER_PATCH) ]]
#endif
#elif defined(USE_TESSELLATION)
[[ patch(triangle, 3) ]]
#endif
    
vertex commonprofile_io commonprofile_vert888B9528C61D5CDB4B5375EA6FF113CC407275DDF4A9915FF19A54A966BC4E3A(
#if !defined(USE_TESSELLATION)
                                           scn_vertex_t                       in                               [[ stage_in ]]
                                           , uint                             scn_vertexID                     [[ vertex_id ]]
#else // USE_TESSELLATION
                                           
#ifdef USE_OPENSUBDIV
#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
                                           PatchInput                         patchInput                       [[ stage_in ]]
#else
                                           OsdVertexBufferSet                 patchInput
#endif
                                           , float2                           patchCoord                       [[ position_in_patch ]]
                                           , uint                             patchID                          [[ patch_id ]]
                                           , constant float&                  osdTessellationLevel             [[ buffer(TESSELLATION_LEVEL_BUFFER_INDEX) ]]
#else // OSD_IS_ADAPTIVE
                                           device unsigned const*             osdIndicesBuffer                 [[ buffer(INDICES_BUFFER_INDEX) ]]
                                           , device osd_packed_vertex const*  osdVertexBuffer                  [[ buffer(VERTEX_BUFFER_INDEX) ]]
                                           , uint                             vertexID                         [[ vertex_id ]]
#endif // OSD_IS_ADAPTIVE
#if defined(OSD_FVAR_WIDTH)
#if OSD_FVAR_USES_MULTIPLE_CHANNELS
                                           , constant uint32_t&               osdFaceVaryingChannelCount       [[ buffer(OSD_FVAR_CHANNELS_CHANNEL_COUNT_INDEX) ]]
                                           , device OsdFVarChannelDesc const* osdFaceVaryingChannelDescriptors [[ buffer(OSD_FVAR_CHANNELS_CHANNEL_DESCRIPTORS_INDEX) ]]
                                           , constant uint32_t&               osdFaceVaryingPatchArrayIndex    [[ buffer(OSD_FVAR_CHANNELS_PATCH_ARRAY_INDEX_BUFFER_INDEX) ]]
                                           , device void const*               osdFaceVaryingChannelsPackedData [[ buffer(OSD_FVAR_CHANNELS_PACKED_DATA_BUFFER_INDEX) ]]
#else
                                           , device float const*              osdFaceVaryingData               [[ buffer(OSD_FVAR_DATA_BUFFER_INDEX) ]]
                                           , device int const*                osdFaceVaryingIndices            [[ buffer(OSD_FVAR_INDICES_BUFFER_INDEX) ]]
#if OSD_IS_ADAPTIVE
                                           , device packed_int3 const*        osdFaceVaryingPatchParams        [[ buffer(OSD_FVAR_PATCHPARAM_BUFFER_INDEX) ]]
                                           , constant packed_int4&            osdFaceVaryingPatchArray         [[ buffer(OSD_FVAR_PATCH_ARRAY_BUFFER_INDEX) ]]
#endif
#endif //OSD_FVAR_USES_MULTIPLE_CHANNELS
#endif //defined(OSD_FVAR_WIDTH)
#else // USE_OPENSUBDIV
                                           scn_patch_t                        in                               [[ stage_in ]]
                                           , float3                           patchCoord                       [[ position_in_patch ]]
#endif // USE_OPENSUBDIV
#endif // USE_TESSELLATION
                                           
#ifdef USE_MULTIPLE_RENDERING
                                           , device SCNSceneBuffer*           scn_frames                       [[ buffer(0) ]]
#else
                                           , constant SCNSceneBuffer&         scn_frame                        [[ buffer(0) ]]
#endif
#ifdef USE_INSTANCING
                                           // we use device here to override the 64Ko limit of constant buffers on NV hardware
                                           , device commonprofile_node*       scn_nodeInstances                [[ buffer(1) ]]
#else
                                           , constant commonprofile_node&     scn_node                         [[ buffer(1) ]]
#endif
#ifdef USE_PER_VERTEX_LIGHTING
                                           , constant scn_light*              scn_lights                       [[ buffer(2) ]]
                                           , constant float4*                 u_shadowKernel
#endif
                                           // used for texture transform and materialShininess in case of perVertexLighting
                                           , constant commonprofile_uniforms& scn_commonprofile
                                           , uint                             scn_instanceID                   [[ instance_id ]]
                                           
#ifdef USE_POINT_RENDERING
                                           // x:pointSize, y:minimumScreenSize, z:maximumScreenSize
                                           , constant float3&                 scn_pointSize
#endif
#ifdef USE_DISPLACEMENT_MAP
#if USE_ARGUMENT_BUFFERS
#define u_displacementTexture           scn_commonprofile.displacementTexture
#define u_displacementTextureSampler    scn_commonprofile.displacementSampler
#else
                                           , texture2d<float>                 u_displacementTexture
                                           , sampler                          u_displacementTextureSampler
#endif //USE_ARGUMENT_BUFFERS
#endif //USE_DISPLACEMENT_MAP
#ifdef USE_VERTEX_EXTRA_ARGUMENTS
                                           
#endif
                                           )
{
    commonprofile_io out;
    
#ifdef USE_MULTIPLE_RENDERING
    
    out.sliceIndex = scn_instanceID % USE_MULTIPLE_RENDERING;
    
    device SCNSceneBuffer& scn_frame = scn_frames[0];
    device SCNSceneBuffer& scn_frame_slice = scn_frames[out.sliceIndex];
#ifdef USE_INSTANCING
    device commonprofile_node& scn_node = scn_nodeInstances[scn_instanceID / USE_MULTIPLE_RENDERING];
#endif
    
#else
    
#ifdef USE_INSTANCING
    device commonprofile_node& scn_node = scn_nodeInstances[scn_instanceID];
#endif
    
#endif
    
    
#ifdef USE_TESSELLATION
    uint scn_vertexID; // we need scn_vertexID if a geometry modifier is used
    scn_vertexID = 0;
#endif
    
    //
    // MARK: Populating the `_geometry` struct
    //
    
    SCNShaderGeometry _geometry;
    
#if !defined(USE_TESSELLATION)
    
    // OPTIM in could be already float4?
    _geometry.position = float4(in.position, 1.f);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
    _geometry.normal = in.normal;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = in.tangent;
#endif
#ifdef NEED_IN_TEXCOORD0
    _geometry.texcoords[0] = in.texcoord0;
#endif
#ifdef NEED_IN_TEXCOORD1
    _geometry.texcoords[1] = in.texcoord1;
#endif
#ifdef NEED_IN_TEXCOORD2
    _geometry.texcoords[2] = in.texcoord2;
#endif
#ifdef NEED_IN_TEXCOORD3
    _geometry.texcoords[3] = in.texcoord3;
#endif
#ifdef NEED_IN_TEXCOORD4
    _geometry.texcoords[4] = in.texcoord4;
#endif
#ifdef NEED_IN_TEXCOORD5
    _geometry.texcoords[5] = in.texcoord5;
#endif
#ifdef NEED_IN_TEXCOORD6
    _geometry.texcoords[6] = in.texcoord6;
#endif
#ifdef NEED_IN_TEXCOORD7
    _geometry.texcoords[7] = in.texcoord7;
#endif
#ifdef HAS_VERTEX_COLOR
    _geometry.color = in.color;
#elif USE_VERTEX_COLOR
    _geometry.color = float4(1.);
#endif
    
#else // USE_TESSELLATION
    
#ifdef USE_OPENSUBDIV
#if OSD_IS_ADAPTIVE
#if USE_STAGE_IN
    int3 patchParam = patchInput.patchParam;
#else
    int3 patchParam = patchInput.patchParamBuffer[patchID];
#endif
    
    int refinementLevel = OsdGetPatchRefinementLevel(patchParam);
    float tessellationLevel = min(osdTessellationLevel, (float)OSD_MAX_TESS_LEVEL) / exp2((float)refinementLevel - 1);
    
    OsdPatchVertex patchVertex = OsdComputePatch(tessellationLevel, patchCoord, patchID, patchInput);
    
#if defined(OSD_FVAR_WIDTH)
    int patchIndex = OsdGetPatchIndex(patchID);
#if OSD_FVAR_USES_MULTIPLE_CHANNELS
    OsdInterpolateFaceVarings(_geometry, patchCoord.xy, patchIndex, osdFaceVaryingChannelCount, osdFaceVaryingChannelDescriptors, osdFaceVaryingPatchArrayIndex, osdFaceVaryingChannelsPackedData);
#else
    OsdInterpolateFaceVarings(_geometry, patchCoord.xy, patchIndex, osdFaceVaryingIndices, osdFaceVaryingData, osdFaceVaryingPatchParams, osdFaceVaryingPatchArray);
#endif
#endif
    
    _geometry.position = float4(patchVertex.position, 1.f);
    
#if defined(USE_NORMAL)
    _geometry.normal = patchVertex.normal;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = float4(patchVertex.tangent, -1.f);
    //_geometry.bitangent = patchVertex.bitangent;
#endif
#if defined(NEED_IN_TEXCOORD0) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[0] = patchVertex.texcoord0;
#endif
#if defined(NEED_IN_TEXCOORD1) && (OSD_TEXCOORD1_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[1] = patchVertex.texcoord1;
#endif
#if defined(NEED_IN_TEXCOORD2) && (OSD_TEXCOORD2_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[2] = patchVertex.texcoord2;
#endif
#if defined(NEED_IN_TEXCOORD3) && (OSD_TEXCOORD3_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[3] = patchVertex.texcoord3;
#endif
#if defined(NEED_IN_TEXCOORD4) && (OSD_TEXCOORD4_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[4] = patchVertex.texcoord4;
#endif
#if defined(NEED_IN_TEXCOORD5) && (OSD_TEXCOORD5_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[5] = patchVertex.texcoord5;
#endif
#if defined(NEED_IN_TEXCOORD6) && (OSD_TEXCOORD6_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[6] = patchVertex.texcoord6;
#endif
#if defined(NEED_IN_TEXCOORD7) && (OSD_TEXCOORD7_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[7] = patchVertex.texcoord7;
#endif
#if defined(HAS_VERTEX_COLOR) && (OSD_COLOR_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.color = patchVertex.color;
#endif
    
#else // OSD_IS_ADAPTIVE
    
#if OSD_PATCH_QUADS
    const uint primitiveIndex = vertexID / 6;
#ifdef USE_NORMAL
    float3 p0 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 0]].position;
    float3 p1 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 1]].position;
    float3 p2 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + 2]].position;
    float3 normal = normalize(cross(p2 - p1, p0 - p1));
#endif
    const uint triangleIndices[6] = { 0, 1, 2, 0, 2, 3 };
    const uint quadVertexIndex = triangleIndices[vertexID % 6];
    osd_packed_vertex osdVertex = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 4 + quadVertexIndex]];
#elif OSD_PATCH_TRIANGLES
    const uint primitiveIndex = vertexID / 3;
#ifdef USE_NORMAL
    float3 p0 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 0]].position;
    float3 p1 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 1]].position;
    float3 p2 = osdVertexBuffer[osdIndicesBuffer[primitiveIndex * 3 + 2]].position;
    float3 normal = normalize(cross(p2 - p1, p0 - p1));
#endif
    osd_packed_vertex osdVertex = osdVertexBuffer[osdIndicesBuffer[vertexID]];
#endif
    
    float3 position = osdVertex.position;
    
#if defined(OSD_FVAR_WIDTH)
    int patchIndex = OsdGetPatchIndex(primitiveIndex);
#if OSD_PATCH_QUADS
    float2 quadUVs[4] = { float2(0,0), float2(1,0), float2(1,1), float2(0,1) };
#if OSD_FVAR_USES_MULTIPLE_CHANNELS
    OsdInterpolateFaceVarings(_geometry, quadUVs[quadVertexIndex], patchIndex, osdFaceVaryingChannelCount, osdFaceVaryingChannelDescriptors, osdFaceVaryingPatchArrayIndex, osdFaceVaryingChannelsPackedData);
#else
    OsdInterpolateFaceVarings(_geometry, quadUVs[quadVertexIndex], patchIndex, osdFaceVaryingIndices, osdFaceVaryingData);
#endif
#elif OSD_PATCH_TRIANGLES
    //TODO
#endif
#endif //defined(OSD_FVAR_WIDTH)
    
#if defined(NEED_IN_TEXCOORD0) && (OSD_TEXCOORD0_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[0] = osdVertex.texcoord0;
#endif
#if defined(NEED_IN_TEXCOORD1) && (OSD_TEXCOORD1_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[1] = osdVertex.texcoord1;
#endif
#if defined(NEED_IN_TEXCOORD2) && (OSD_TEXCOORD2_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[2] = osdVertex.texcoord2;
#endif
#if defined(NEED_IN_TEXCOORD3) && (OSD_TEXCOORD3_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[3] = osdVertex.texcoord3;
#endif
#if defined(NEED_IN_TEXCOORD4) && (OSD_TEXCOORD4_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[4] = osdVertex.texcoord4;
#endif
#if defined(NEED_IN_TEXCOORD5) && (OSD_TEXCOORD5_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[5] = osdVertex.texcoord5;
#endif
#if defined(NEED_IN_TEXCOORD6) && (OSD_TEXCOORD6_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[6] = osdVertex.texcoord6;
#endif
#if defined(NEED_IN_TEXCOORD7) && (OSD_TEXCOORD7_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.texcoords[7] = osdVertex.texcoord7;
#endif
#if defined(HAS_VERTEX_COLOR) && (OSD_COLOR_INTERPOLATION_MODE == OSD_PRIMVAR_INTERPOLATION_MODE_USER_VARYING)
    _geometry.color = osdVertex.color;
#endif
    
    _geometry.position = float4(position, 1.f);
#ifdef USE_NORMAL
    _geometry.normal = normal;
#endif
    
#endif // OSD_IS_ADAPTIVE
    
#else // USE_OPENSUBDIV
    
    //
    // MARK: Geometry smooting
    //
    
#if defined(TESSELLATION_SMOOTHING_MODE_PN_TRIANGLE) || defined(TESSELLATION_SMOOTHING_MODE_PHONG)
    float3 P0 = in.controlPoints[0].position;
    float3 P1 = in.controlPoints[1].position;
    float3 P2 = in.controlPoints[2].position;
    float3 N0 = in.controlPoints[0].normal;
    float3 N1 = in.controlPoints[1].normal;
    float3 N2 = in.controlPoints[2].normal;
#if defined(TESSELLATION_SMOOTHING_MODE_PN_TRIANGLE)
    float3 position, normal;
    scn_smooth_geometry_pn_triangle(position, normal, patchCoord, P0, P1, P2, N0, N1, N2);
#elif defined(TESSELLATION_SMOOTHING_MODE_PHONG)
    float3 position, normal;
    scn_smooth_geometry_phong(position, normal, patchCoord, P0, P1, P2, N0, N1, N2);
#endif
    _geometry.position = float4(position, 1.f);
#ifdef USE_NORMAL
    _geometry.normal = normal;
#endif
#else // GEOMETRY_SMOOTHING
    // OPTIM in could be already float4?
    _geometry.position = float4(scn::barycentric_mix(in.controlPoints[0].position, in.controlPoints[1].position, in.controlPoints[2].position, patchCoord), 1.f);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
    _geometry.normal = normalize(scn::barycentric_mix(in.controlPoints[0].normal, in.controlPoints[1].normal, in.controlPoints[2].normal, patchCoord));
#endif
#endif // GEOMETRY_SMOOTHING
    
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
    _geometry.tangent = normalize(scn::barycentric_mix(in.controlPoints[0].tangent, in.controlPoints[1].tangent, in.controlPoints[2].tangent, patchCoord));
#endif
#ifdef NEED_IN_TEXCOORD0
    _geometry.texcoords[0] = scn::barycentric_mix(in.controlPoints[0].texcoord0, in.controlPoints[1].texcoord0, in.controlPoints[2].texcoord0, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD1
    _geometry.texcoords[1] = scn::barycentric_mix(in.controlPoints[0].texcoord1, in.controlPoints[1].texcoord1, in.controlPoints[2].texcoord1, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD2
    _geometry.texcoords[2] = scn::barycentric_mix(in.controlPoints[0].texcoord2, in.controlPoints[1].texcoord2, in.controlPoints[2].texcoord2, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD3
    _geometry.texcoords[3] = scn::barycentric_mix(in.controlPoints[0].texcoord3, in.controlPoints[1].texcoord3, in.controlPoints[2].texcoord3, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD4
    _geometry.texcoords[4] = scn::barycentric_mix(in.controlPoints[0].texcoord4, in.controlPoints[1].texcoord4, in.controlPoints[2].texcoord4, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD5
    _geometry.texcoords[5] = scn::barycentric_mix(in.controlPoints[0].texcoord5, in.controlPoints[1].texcoord5, in.controlPoints[2].texcoord5, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD6
    _geometry.texcoords[6] = scn::barycentric_mix(in.controlPoints[0].texcoord6, in.controlPoints[1].texcoord6, in.controlPoints[2].texcoord6, patchCoord);
#endif
#ifdef NEED_IN_TEXCOORD7
    _geometry.texcoords[7] = scn::barycentric_mix(in.controlPoints[0].texcoord7, in.controlPoints[1].texcoord7, in.controlPoints[2].texcoord7, patchCoord);
#endif
#ifdef HAS_VERTEX_COLOR
    _geometry.color = scn::barycentric_mix(in.controlPoints[0].color, in.controlPoints[1].color, in.controlPoints[2].color, patchCoord);
#elif USE_VERTEX_COLOR
    _geometry.color = float4(1.);
#endif
    
#endif // USE_OPENSUBDIV
    
#endif // USE_TESSELLATION
    
#ifdef USE_POINT_RENDERING
    _geometry.pointSize = scn_pointSize.x;
#endif
    
#ifdef USE_TEXCOORD
    
#endif
    
#ifdef USE_DISPLACEMENT_MAP
    applyDisplacement(u_displacementTexture, u_displacementTextureSampler, _displacementTexcoord, _geometry, scn_commonprofile);
#endif
    
    //
    // MARK: Skinning
    //
    
#ifdef USE_SKINNING
#if !defined(USE_TESSELLATION)
    {
        float3 pos = 0.f;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        float3 nrm = 0.f;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        float3 tgt = 0.f;
#endif
        for (int i = 0; i < MAX_BONE_INFLUENCES; ++i) {
#if MAX_BONE_INFLUENCES == 1
            float weight = 1.f;
#else
            float weight = in.skinningWeights[i];
            if (weight <= 0.f)
                continue;
            
#endif
            int idx = int(in.skinningJoints[i]) * 3;
            float4x4 jointMatrix = float4x4(scn_node.skinningJointMatrices[idx],
                                            scn_node.skinningJointMatrices[idx+1],
                                            scn_node.skinningJointMatrices[idx+2],
                                            float4(0., 0., 0., 1.));
            
            pos += (_geometry.position * jointMatrix).xyz * weight;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
            nrm += _geometry.normal * scn::mat3(jointMatrix) * weight;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
            tgt += _geometry.tangent.xyz * scn::mat3(jointMatrix) * weight;
#endif
        }
        
        _geometry.position.xyz = pos;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        _geometry.normal = nrm;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        _geometry.tangent.xyz = tgt;
#endif
    }
    
#else // USE_TESSELLATION
    
#if !defined(USE_OPENSUBDIV)
    {
        float3 pos[3] = {0.f, 0.f, 0.f};
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        float3 nrm[3] = {0.f, 0.f, 0.f};
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        float3 tgt[3] = {0.f, 0.f, 0.f};
#endif
        for (int controlPointIndex = 0; controlPointIndex < 3; ++controlPointIndex) {
            for (int i = 0; i < MAX_BONE_INFLUENCES; ++i) {
#if MAX_BONE_INFLUENCES == 1
                float weight = 1.f;
#else
                float weight = in.controlPoints[controlPointIndex].skinningWeights[i];
                if (weight <= 0.f)
                    continue;
                
#endif
                int idx = int(in.controlPoints[controlPointIndex].skinningJoints[i]) * 3;
                float4x4 jointMatrix = float4x4(scn_node.skinningJointMatrices[idx],
                                                scn_node.skinningJointMatrices[idx+1],
                                                scn_node.skinningJointMatrices[idx+2],
                                                float4(0., 0., 0., 1.));
                
                pos[controlPointIndex] += (_geometry.position * jointMatrix).xyz * weight;
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
                nrm[controlPointIndex] += _geometry.normal * scn::mat3(jointMatrix) * weight;
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
                tgt[controlPointIndex] += _geometry.tangent.xyz * scn::mat3(jointMatrix) * weight;
#endif
            }
        }
        
        _geometry.position.xyz = scn::barycentric_mix(pos[0], pos[1], pos[2], patchCoord);
#if defined(USE_NORMAL) && defined(HAS_NORMAL)
        _geometry.normal = scn::barycentric_mix(nrm[0], nrm[1], nrm[2], patchCoord);
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
        _geometry.tangent.xyz = scn::barycentric_mix(tgt[0], tgt[1], tgt[2], patchCoord);
#endif
    }
#endif // !defined(USE_OPENSUBDIV)
#endif // USE_TESSELLATION
#endif // USE_SKINNING
    
    
#ifdef USE_DISPLACEMENT_MAP
    out.displacementTexcoord = _displacementTexcoord;
#endif
    
    //
    // MARK: Geometry shader modifier
    //
    
#ifdef USE_GEOMETRY_MODIFIER
    // DoGeometryModifier START
    
    // DoGeometryModifier END
#endif
    
    //
    // MARK: Populating the `_surface` struct
    //
    
    // Transform the geometry elements in view space
#if defined(USE_POSITION) || (defined(USE_NORMAL) && defined(HAS_OR_GENERATES_NORMAL)) || defined(USE_TANGENT) || defined(USE_BITANGENT) || defined(USE_INSTANCING)
    SCNShaderSurface _surface;
#endif
#if defined(USE_POSITION) || defined(USE_INSTANCING)
#ifdef USE_MULTIPLE_RENDERING
    _surface.position = (scn_frame.viewTransform * scn_node.modelTransform * _geometry.position).xyz;
#else
    _surface.position = (scn_node.modelViewTransform * _geometry.position).xyz;
#endif
#endif
#if defined(USE_NORMAL) && defined(HAS_OR_GENERATES_NORMAL)
#ifdef USE_MULTIPLE_RENDERING
#ifdef HINT_UNIFORM_SCALE
    _surface.normal = (scn_frame.viewTransform * scn_node.modelTransform * float4(_geometry.normal,0.)).xyz;
#else
    _surface.normal = normalize( (scn_frame.inverseTransposeViewTransform * scn_node.modelTransform * float4(_geometry.normal,0.)).xyz );
#endif
#else
    float3x3 nrmTransform = scn::mat3(scn_node.modelViewTransform);
#ifdef HINT_UNIFORM_SCALE
    _surface.normal = nrmTransform * _geometry.normal;
#else
    float3 invScaleSquared = 1.f / float3(length_squared(nrmTransform[0]), length_squared(nrmTransform[1]), length_squared(nrmTransform[2]));
    _surface.normal = normalize(nrmTransform * (_geometry.normal * invScaleSquared));
#endif
#endif
#endif
#if defined(USE_TANGENT) || defined(USE_BITANGENT)
#ifdef USE_MULTIPLE_RENDERING
    _surface.tangent = normalize( (scn_frame.viewTransform * scn_node.modelTransform * float4(_geometry.tangent.xyz, 0.f)).xyz );
#else
    _surface.tangent = normalize(scn::mat3(scn_node.modelViewTransform) * _geometry.tangent.xyz);
#endif
    _surface.bitangent = _geometry.tangent.w * cross(_surface.tangent, _surface.normal); // no need to renormalize since tangent and normal should be orthogonal
    // old code : _surface.bitangent =  normalize(cross(_surface.normal,_surface.tangent));
#endif
    
    //if USE_VIEW is 2 we may also need to set _surface.view. todo: make USE_VIEW a mask
#ifdef USE_VIEW
    _surface.view = normalize(-_surface.position);
#endif
    
    //
    // MARK: Per-vertex lighting
    //
    
#ifdef USE_PER_VERTEX_LIGHTING
    // Lighting
    SCNShaderLightingContribution _lightingContribution(_surface, out);
    _lightingContribution.diffuse = 0.;
  #ifdef USE_SPECULAR
    _lightingContribution.specular = 0.;
    _surface.shininess = scn_commonprofile.materialShininess;
  #endif
    
    out.diffuse = _lightingContribution.diffuse;
  #ifdef USE_SPECULAR
    out.specular = _lightingContribution.specular;
  #endif
#endif
#if defined(USE_POSITION) && (USE_POSITION == 2)
    out.position = _surface.position;
#endif
#if defined(USE_NORMAL) && (USE_NORMAL == 2) && defined(HAS_OR_GENERATES_NORMAL)
    out.normal = _surface.normal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    out.tangent = _surface.tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    out.bitangent = _surface.bitangent;
#endif
#ifdef USE_VERTEX_COLOR
    out.vertexColor = _geometry.color;
#endif
#ifdef USE_TEXCOORD
    
#endif
    
    //
    // MARK: Determining the fragment position
    //
    
#if defined(USE_POSITION) || defined(USE_INSTANCING)
#ifdef USE_MULTIPLE_RENDERING
    out.fragmentPosition = scn_frame_slice.viewProjectionTransform * scn_node.modelTransform * _geometry.position;
#else
    out.fragmentPosition = scn_frame.projectionTransform * float4(_surface.position, 1.);
#endif
#elif defined(USE_MODELVIEWPROJECTIONTRANSFORM) // this means that the geometry are still in model space : we can transform it directly to NDC space
#ifdef USE_MULTIPLE_RENDERING
    out.fragmentPosition = scn_frame_slice.viewProjectionTransform * scn_node.modelTransform * _geometry.position;
#else
    out.fragmentPosition = scn_node.modelViewProjectionTransform * _geometry.position;
#endif
#endif
    
#ifdef USE_NODE_OPACITY
    out.nodeOpacity = scn_node.nodeOpacity;
#endif
    
#ifdef USE_POINT_RENDERING
    float screenSize = _geometry.pointSize / out.fragmentPosition.w;
    out.fragmentSize = clamp(screenSize, scn_pointSize.y, scn_pointSize.z);
#endif
    
#ifdef USE_MOTIONBLUR
    float4 lastFrameFragmentPosition = scn_frame.lastFrameViewProjectionTransform * scn_node.lastFrameModelTransform * _geometry.position;
    out.velocity.xy = lastFrameFragmentPosition.xy * float2(1., -1.);
    out.velocity.z = lastFrameFragmentPosition.w;
#endif
    
#ifdef USE_OUTLINE
	out.outlineHash = hash(scn_node.modelTransform[3].xy)+1.f/255.f;
#endif
    
    return out;
}

//
// MARK: - Fragment shader function
//

struct SCNOutput
{
    float4 color [[ color(0) ]];
#ifdef USE_COLOR1_OUTPUT
    half4 color1 [[ color(1) ]];
#endif
#ifdef USE_REFLECTANCE_ROUGHNESS_OUTPUT
    half4 reflectanceRoughnessOutput [[ color(2) ]];
#endif
#ifdef USE_MOTIONBLUR
    half4 motionblur [[ color(3) ]];
#endif
#ifdef USE_NORMALS_OUTPUT
    half4 normals [[ color(4) ]];
#endif
};

fragment SCNOutput commonprofile_frag888B9528C61D5CDB4B5375EA6FF113CC407275DDF4A9915FF19A54A966BC4E3A(commonprofile_io                 in                               [[ stage_in  ]]
                                      , constant commonprofile_uniforms& scn_commonprofile              [[ buffer(0) ]]
#ifdef USE_MULTIPLE_RENDERING
                                      , device SCNSceneBuffer*           scn_frames                     [[ buffer(1) ]]
#else
                                      , constant SCNSceneBuffer&         scn_frame                      [[ buffer(1) ]]
#endif
                                      , constant commonprofile_node&  scn_node                          [[ buffer(2) ]]

#ifdef USE_PER_PIXEL_LIGHTING
                                      , constant scn_light*            scn_lights                       [[ buffer(3) ]]
                                      , constant float4*               u_shadowKernel
#ifdef C3D_SUPPORT_CUBE_ARRAY
                                      , texturecube_array<half>        u_reflectionProbeTexture
#else
                                      , texture2d_array<half>          u_reflectionProbeTexture
#endif
                                      , texture3d<ushort>              u_clusterTexture
#ifdef C3D_USE_TEXTURE_FOR_LIGHT_INDICES
                                      , texture1d<ushort>              u_lightIndicesTexture
#else
                                      , constant C3DLightIndexType*    u_lightIndicesBuffer
#endif
#endif
#if USE_ARGUMENT_BUFFERS

#define u_emissionTexture               scn_commonprofile.emissionTexture
#define u_emissionTextureSampler        scn_commonprofile.emissionSampler
#define u_ambientTexture                scn_commonprofile.ambientTexture
#define u_ambientTextureSampler         scn_commonprofile.ambientSampler
#define u_diffuseTexture                scn_commonprofile.diffuseTexture
#define u_diffuseTextureSampler         scn_commonprofile.diffuseSampler
#define u_specularTexture               scn_commonprofile.specularTexture
#define u_specularTextureSampler        scn_commonprofile.specularSampler
#define u_reflectiveTexture             scn_commonprofile.reflectiveTexture
#define u_reflectiveTextureSampler      scn_commonprofile.reflectiveSampler
#define u_transparentTexture            scn_commonprofile.transparentTexture
#define u_transparentTextureSampler     scn_commonprofile.transparentSampler
#define u_multiplyTexture               scn_commonprofile.multiplyTexture
#define u_multiplyTextureSampler        scn_commonprofile.multiplySampler
#define u_normalTexture                 scn_commonprofile.normalTexture
#define u_normalTextureSampler          scn_commonprofile.normalSampler
#define u_selfIlluminationTexture       scn_commonprofile.selfIlluminationTexture
#define u_selfIlluminationTextureSampler scn_commonprofile.selfIlluminationSampler
#define u_metalnessTexture              scn_commonprofile.metalnessTexture
#define u_metalnessTextureSampler       scn_commonprofile.metalnessSampler
#define u_roughnessTexture              scn_commonprofile.roughnessTexture
#define u_roughnessTextureSampler       scn_commonprofile.roughnessSampler

#else
#ifdef USE_EMISSION_MAP
                                      , texture2d<float>              u_emissionTexture
                                      , sampler                       u_emissionTextureSampler
#endif
#ifdef USE_AMBIENT_MAP
                                      , texture2d<float>              u_ambientTexture
                                      , sampler                       u_ambientTextureSampler
#endif
#ifdef USE_DIFFUSE_MAP
                                      , texture2d<float>              u_diffuseTexture
                                      , sampler                       u_diffuseTextureSampler
#endif
#ifdef USE_SPECULAR_MAP
                                      , texture2d<float>              u_specularTexture
                                      , sampler                       u_specularTextureSampler
#endif
#ifdef USE_REFLECTIVE_MAP
                                      , texture2d<float>              u_reflectiveTexture
                                      , sampler                       u_reflectiveTextureSampler
#elif defined(USE_REFLECTIVE_CUBEMAP)
                                      , texturecube<float>            u_reflectiveTexture
                                      , sampler                       u_reflectiveTextureSampler
#endif
#ifdef USE_TRANSPARENT_MAP
                                      , texture2d<float>              u_transparentTexture
                                      , sampler                       u_transparentTextureSampler
#endif
#ifdef USE_MULTIPLY_MAP
                                      , texture2d<float>              u_multiplyTexture
                                      , sampler                       u_multiplyTextureSampler
#endif
#ifdef USE_NORMAL_MAP
                                      , texture2d<float>              u_normalTexture
                                      , sampler                       u_normalTextureSampler
#endif
#ifdef USE_SELFILLUMINATION_MAP
                                      , texture2d<float>              u_selfIlluminationTexture
                                      , sampler                       u_selfIlluminationTextureSampler
#endif
#ifdef USE_DISPLACEMENT_MAP
                                      , texture2d<float>              u_displacementTexture
                                      , sampler                       u_displacementTextureSampler
#endif
#ifdef USE_PBR
#ifdef USE_METALNESS_MAP
                                      , texture2d<float>              u_metalnessTexture
                                      , sampler                       u_metalnessTextureSampler
#endif
#ifdef USE_ROUGHNESS_MAP
                                      , texture2d<float>              u_roughnessTexture
                                      , sampler                       u_roughnessTextureSampler
#endif
#endif // USE_PBR
#endif // USE_ARGUMENT_BUFFERS
#ifdef USE_PBR
                                      , texturecube<float>            u_radianceTexture
                                      , texture2d<float>              u_specularDFGTexture
#if !defined(USE_SELFILLUMINATION_MAP)
                                      , texturecube<float>            u_irradianceTexture
#endif
#endif // USE_PBR
#ifdef USE_SSAO
                                      , texture2d<float>              u_ssaoTexture
#endif
#ifdef USE_FRAGMENT_EXTRA_ARGUMENTS
                                      
#endif
#if defined(USE_DOUBLE_SIDED)
                                      , bool                          isFrontFacing                    [[front_facing]]
#endif
#ifdef USE_POINT_RENDERING
                                      , float2                        pointCoord                       [[point_coord]]
#endif
                                      )
{
#ifdef USE_MULTIPLE_RENDERING
    device SCNSceneBuffer& scn_frame = scn_frames[0];
    device SCNSceneBuffer& scn_frame_slice = scn_frames[in.sliceIndex];
#endif
    SCNOutput _output;

    //
    // MARK: Populating the `_surface` struct
    //
    
    SCNShaderSurface _surface;
#ifdef USE_TEXCOORD
    
#endif
    _surface.ambientOcclusion = 1.f; // default to no AO
#ifdef USE_AMBIENT_MAP
    #ifdef USE_AMBIENT_AS_AMBIENTOCCLUSION
        _surface.ambientOcclusion = u_ambientTexture.sample(u_ambientTextureSampler, _surface.ambientTexcoord).r;
        #ifdef USE_AMBIENT_INTENSITY
            _surface.ambientOcclusion = saturate(mix(1.f, _surface.ambientOcclusion, scn_commonprofile.ambientIntensity));
        #endif
    #else // AMBIENT_MAP
        _surface.ambient = u_ambientTexture.sample(u_ambientTextureSampler, _surface.ambientTexcoord);
        #ifdef USE_AMBIENT_INTENSITY
            _surface.ambient *= scn_commonprofile.ambientIntensity;
        #endif
    #endif // USE_AMBIENT_AS_AMBIENTOCCLUSION
#if defined(USE_AMBIENT_TEXTURE_COMPONENT)
    _surface.ambient = colorFromMask(_surface.ambient, USE_AMBIENT_TEXTURE_COMPONENT).r;
#endif

#elif defined(USE_AMBIENT_COLOR)
    _surface.ambient = scn_commonprofile.ambientColor;
#elif defined(USE_AMBIENT)
    _surface.ambient = float4(0.);
#endif
#if defined(USE_AMBIENT) && defined(USE_VERTEX_COLOR)
    _surface.ambient *= in.vertexColor;
#endif
#if  defined(USE_SSAO)
    _surface.ambientOcclusion *= u_ssaoTexture.sample( sampler(filter::linear), in.fragmentPosition.xy * scn_frame.inverseResolution.xy ).x;
#endif
    
#ifdef USE_DIFFUSE_MAP
    _surface.diffuse = u_diffuseTexture.sample(u_diffuseTextureSampler, _surface.diffuseTexcoord);
#if defined(USE_DIFFUSE_TEXTURE_COMPONENT)
    _surface.diffuse = colorFromMask(_surface.diffuse, USE_DIFFUSE_TEXTURE_COMPONENT);
#endif
#ifdef USE_DIFFUSE_INTENSITY
    _surface.diffuse.rgb *= scn_commonprofile.diffuseIntensity;
#endif
#elif defined(USE_DIFFUSE_COLOR)
    _surface.diffuse = scn_commonprofile.diffuseColor;
#else
    _surface.diffuse = float4(0.f,0.f,0.f,1.f);
#endif
#if defined(USE_DIFFUSE) && defined(USE_VERTEX_COLOR)
    _surface.diffuse.rgb    *= in.vertexColor.rgb;
    _surface.diffuse        *= in.vertexColor.a; // vertex color are not premultiplied to allow interpolation
#endif
#ifdef USE_SPECULAR_MAP
    _surface.specular = u_specularTexture.sample(u_specularTextureSampler, _surface.specularTexcoord);
#if defined(USE_SPECULAR_TEXTURE_COMPONENT)
    _surface.specular = colorFromMask(_surface.specular, USE_SPECULAR_TEXTURE_COMPONENT);
#endif
#ifdef USE_SPECULAR_INTENSITY
    _surface.specular *= scn_commonprofile.specularIntensity;
#endif
#elif defined(USE_SPECULAR_COLOR)
    _surface.specular = scn_commonprofile.specularColor;
#elif defined(USE_SPECULAR)
    _surface.specular = float4(0.f);
#endif
#ifdef USE_EMISSION_MAP
    _surface.emission = u_emissionTexture.sample(u_emissionTextureSampler, _surface.emissionTexcoord);
#if defined(USE_EMISSION_TEXTURE_COMPONENT)
    _surface.emission = colorFromMask(_surface.emission, USE_EMISSION_TEXTURE_COMPONENT);
#endif
#ifdef USE_EMISSION_INTENSITY
    _surface.emission *= scn_commonprofile.emissionIntensity;
#endif
#elif defined(USE_EMISSION_COLOR)
    _surface.emission = scn_commonprofile.emissionColor;
#elif defined(USE_EMISSION)
    _surface.emission = float4(0.);
#endif
#ifdef USE_SELFILLUMINATION_MAP
    _surface.selfIllumination = u_selfIlluminationTexture.sample(u_selfIlluminationTextureSampler, _surface.selfIlluminationTexcoord);
#if defined(USE_SELFILLUMINATION_TEXTURE_COMPONENT)
    _surface.selfIllumination = colorFromMask(_surface.selfIllumination, USE_SELFILLUMINATION_TEXTURE_COMPONENT);
#endif
#ifdef USE_SELFILLUMINATION_INTENSITY
    _surface.selfIllumination *= scn_commonprofile.selfIlluminationIntensity;
#endif
#elif defined(USE_SELFILLUMINATION_COLOR)
    _surface.selfIllumination = scn_commonprofile.selfIlluminationColor;
#elif defined(USE_SELFILLUMINATION)
    _surface.selfIllumination = float4(0.);
#endif
#ifdef USE_MULTIPLY_MAP
    _surface.multiply = u_multiplyTexture.sample(u_multiplyTextureSampler, _surface.multiplyTexcoord);
#if defined(USE_MULTIPLY_TEXTURE_COMPONENT)
    _surface.multiply = colorFromMask(_surface.multiply, USE_MULTIPLY_TEXTURE_COMPONENT);
#endif
#ifdef USE_MULTIPLY_INTENSITY
    _surface.multiply = mix(float4(1.), _surface.multiply, scn_commonprofile.multiplyIntensity);
#endif
#elif defined(USE_MULTIPLY_COLOR)
    _surface.multiply = scn_commonprofile.multiplyColor;
#elif defined(USE_MULTIPLY)
    _surface.multiply = float4(1.);
#endif
#ifdef USE_TRANSPARENT_MAP
    _surface.transparent = u_transparentTexture.sample(u_transparentTextureSampler, _surface.transparentTexcoord);
#if defined(USE_TRANSPARENT_TEXTURE_COMPONENT)
    _surface.transparent = colorFromMask(_surface.transparent, USE_TRANSPARENT_TEXTURE_COMPONENT);
#endif
#ifdef USE_TRANSPARENT_INTENSITY
    _surface.transparent *= scn_commonprofile.transparentIntensity;
#endif
#elif defined(USE_TRANSPARENT_COLOR)
    _surface.transparent = scn_commonprofile.transparentColor;
#elif defined(USE_TRANSPARENT)
    _surface.transparent = float4(1.f);
#endif
    
#ifdef USE_METALNESS_MAP
#if defined(USE_METALNESS_TEXTURE_COMPONENT)
    _surface.metalness = colorFromMask(u_metalnessTexture.sample(u_metalnessTextureSampler, _surface.metalnessTexcoord), USE_METALNESS_TEXTURE_COMPONENT).r;
#else
    _surface.metalness = u_metalnessTexture.sample(u_metalnessTextureSampler, _surface.metalnessTexcoord).r;
#endif
#ifdef USE_METALNESS_INTENSITY
    _surface.metalness *= scn_commonprofile.metalnessIntensity;
#endif
#elif defined(USE_METALNESS_COLOR)
    _surface.metalness = scn_commonprofile.metalness;
#else
    _surface.metalness = 0.f;
#endif
    
#ifdef USE_ROUGHNESS_MAP
#if defined(USE_ROUGHNESS_TEXTURE_COMPONENT)
    _surface.roughness = colorFromMask(u_roughnessTexture.sample(u_roughnessTextureSampler, _surface.roughnessTexcoord), USE_ROUGHNESS_TEXTURE_COMPONENT).r;
#else
    _surface.roughness = u_roughnessTexture.sample(u_roughnessTextureSampler, _surface.roughnessTexcoord).r;
#endif
#ifdef USE_ROUGHNESS_INTENSITY
    _surface.roughness *= scn_commonprofile.roughnessIntensity;
#endif
#elif defined(USE_ROUGHNESS_COLOR)
    _surface.roughness = scn_commonprofile.roughness;
#else
    _surface.roughness = 0.f;
#endif
#if (defined USE_POSITION) && (USE_POSITION == 2)
    _surface.position = in.position;
#endif
#if (defined USE_NORMAL) && (USE_NORMAL == 2)
#if defined(HAS_NORMAL) || defined(USE_OPENSUBDIV)
#ifdef USE_DOUBLE_SIDED
    _surface.geometryNormal = normalize(in.normal.xyz) * (isFrontFacing ? 1.f : -1.f );
#else
    _surface.geometryNormal = normalize(in.normal.xyz);
#endif
#else // need to generate the normal from the derivatives
    _surface.geometryNormal = normalize( cross(dfdy( _surface.position ), dfdx( _surface.position ) ));
#endif
    _surface.normal = _surface.geometryNormal;
#endif
#if defined(USE_TANGENT) && (USE_TANGENT == 2)
    _surface.tangent = in.tangent;
#endif
#if defined(USE_BITANGENT) && (USE_BITANGENT == 2)
    _surface.bitangent = in.bitangent;
#endif
#if (defined USE_VIEW) && (USE_VIEW == 2)
    _surface.view = normalize(-in.position);
#endif
#if defined(USE_NORMAL_MAP)
    {
        float3x3 ts2vs = float3x3(_surface.tangent, _surface.bitangent, _surface.normal);
#ifdef USE_NORMAL_MAP
#if defined(USE_NORMAL_TEXTURE_COMPONENT)
        _surface._normalTS.xy = colorFromMask(u_normalTexture.sample(u_normalTextureSampler, _surface.normalTexcoord), USE_NORMAL_TEXTURE_COMPONENT).rg * 2.f - 1.f;
        _surface._normalTS.z = sqrt(1.f - saturate(length_squared(_surface._normalTS.xy)));
#else
        _surface._normalTS = u_normalTexture.sample(u_normalTextureSampler, _surface.normalTexcoord).rgb;
        _surface._normalTS = _surface._normalTS * 2.f - 1.f;
#endif
#ifdef USE_NORMAL_INTENSITY
        _surface._normalTS = mix(float3(0.f, 0.f, 1.f), _surface._normalTS, scn_commonprofile.normalIntensity);
#endif
#else
        _surface._normalTS = float3(0.f, 0.f, 1.f);
#endif
        _surface.normal.rgb = normalize(ts2vs * _surface._normalTS.xyz );
    }
#else
    _surface._normalTS = float3(0.f, 0.f, 1.f);
#endif
#ifdef USE_REFLECTIVE_MAP
    float3 refl = reflect( -_surface.view, _surface.normal );
    float m = 2.f * sqrt( refl.x*refl.x + refl.y*refl.y + (refl.z+1.f)*(refl.z+1.f));
    _surface.reflective = u_reflectiveTexture.sample(u_reflectiveTextureSampler, float2(float2(refl.x,-refl.y) / m) + 0.5f);
#if defined(USE_REFLECTIVE_TEXTURE_COMPONENT)
    _surface.reflective = colorFromMask(_surface.reflective, USE_REFLECTIVE_TEXTURE_COMPONENT).r;
#endif
#ifdef USE_REFLECTIVE_INTENSITY
    _surface.reflective *= scn_commonprofile.reflectiveIntensity;
#endif
#elif defined(USE_REFLECTIVE_CUBEMAP)
    float3 refl = reflect( _surface.position, _surface.normal );
    _surface.reflective = u_reflectiveTexture.sample(u_reflectiveTextureSampler, scn::mat4_mult_float3(scn_frame.viewToCubeTransform, refl)); // sample the cube map in world space
#ifdef USE_REFLECTIVE_INTENSITY
    _surface.reflective *= scn_commonprofile.reflectiveIntensity;
#endif
#elif defined(USE_REFLECTIVE_COLOR)
    _surface.reflective = scn_commonprofile.reflectiveColor;
#elif defined(USE_REFLECTIVE)
    _surface.reflective = float4(0.);
#endif
#ifdef USE_FRESNEL
    _surface.fresnel = scn_commonprofile.fresnel.x + scn_commonprofile.fresnel.y * pow(1.f - saturate(dot(_surface.view, _surface.normal)), scn_commonprofile.fresnel.z);
    _surface.reflective *= _surface.fresnel;
#endif
#ifdef USE_SHININESS
    _surface.shininess = scn_commonprofile.materialShininess;
#endif
    
    //
    // MARK: Surface shader modifier
    //
    
#ifdef USE_SURFACE_MODIFIER
    // DoSurfaceModifier START
    
    // DoSurfaceModifier END
#endif
    
    //
    // MARK: Lighting
    //
    
    SCNShaderLightingContribution _lightingContribution(_surface, in);
#ifdef USE_LIGHT_MODIFIER
    
#endif
#ifdef USE_AMBIENT_LIGHTING
    _lightingContribution.ambient = scn_frame.ambientLightingColor.rgb;
#endif
#ifdef USE_LIGHTING
#ifdef USE_PER_PIXEL_LIGHTING
#ifdef USE_CLUSTERED_LIGHTING
    uint3 clusterIndex;
    clusterIndex.xy = uint2(in.fragmentPosition.xy * scn_frame.clusterScale.xy); // TODO Multiple rendering
    clusterIndex.z = in.position.z * scn_frame.clusterScale.z + scn_frame.clusterScale.w; // scale/bias
    
    // x:offset y:spot<<8|omni z:probe w:????
    ushort4 cluster_offset_count = u_clusterTexture.read(clusterIndex);
    int lid = cluster_offset_count.x;
#endif

#ifdef USE_PBR
    _lightingContribution.prepareForPBR(u_specularDFGTexture, scn_commonprofile.selfIlluminationOcclusion);
    
    // Irradiance
#ifdef USE_SELFILLUMINATION
    _lightingContribution.add_irradiance_from_selfIllum();
#else
#ifdef USE_PROBES_LIGHTING // Irradiance SH
    _lightingContribution.add_global_irradiance_from_sh(scn_frame.viewToCubeTransform, scn_node.shCoefficients);
#else
    _lightingContribution.add_global_irradiance_probe(u_irradianceTexture, scn_frame.viewToCubeTransform, scn_frame.environmentIntensity);
#endif
#endif

    // Radiance
#ifndef DISABLE_SPECULAR
#ifdef C3D_USE_REFLECTION_PROBES
    int probe_count = (cluster_offset_count.z & 0xff);
    for (int i = 0 ; i < probe_count; ++i, ++lid) {
        _lightingContribution.add_local_probe(scn_lights[LightIndex(lid)], u_reflectionProbeTexture);
    }
#if PROBES_NORMALIZATION
#if PROBES_OUTER_BLENDING
    _lightingContribution.specular += _lightingContribution.probesWeightedSum.rgb / max(1.f, _lightingContribution.probesWeightedSum.a);
#else
    _lightingContribution.specular += _lightingContribution.probesWeightedSum.rgb / _lightingContribution.probesWeightedSum.a;
#endif
    float globalFactor = saturate(1.f - _lightingContribution.probesWeightedSum.a);
#else
    float globalFactor = _lightingContribution.probeRadianceRemainingFactor;
#endif
    _lightingContribution.add_global_probe(scn_frame.viewToCubeTransform, globalFactor * scn_frame.environmentIntensity,
                                           u_reflectionProbeTexture);
#else // Global Probe
    _lightingContribution.add_global_probe(u_radianceTexture, scn_frame.viewToCubeTransform, scn_frame.environmentIntensity);
#endif // C3D_USE_REFLECTION_PROBES
#endif // DISABLE_SPECULAR

#endif // USE_PBR
    #if DEBUG_PIXEL
        switch (DEBUG_PIXEL) {
            case 1: _output.color = float4(_surface.normal * 0.5f + 0.5f, 1.f); break;
            case 2: _output.color = float4(_surface.geometryNormal * 0.5f + 0.5f, 1.f); break;
            case 3: _output.color = float4(_surface.tangent * 0.5f + 0.5f, 1.f); break;
            case 4: _output.color = float4(_surface.bitangent * 0.5f + 0.5f, 1.f); break;
            case 5: _output.color = float4(_surface.diffuse.rgb, 1.f); break;
            case 6: _output.color = float4(float3(_surface.roughness), 1.f); break;
            case 7: _output.color = float4(float3(_surface.metalness), 1.f); break;
            case 8: _output.color = float4(float3(_surface.ambientOcclusion), 1.f); break;
            default: break;
        }
        return _output;
    #endif

    #ifdef USE_CLUSTERED_LIGHTING
        // Omni lights
        int omni_count = cluster_offset_count.y & 0xff;
        for (int i = 0 ; i < omni_count; ++i, ++lid) {
            _lightingContribution.add_local_omni(scn_lights[LightIndex(lid)]);
        }

        // Spot lights
        int spot_count = (cluster_offset_count.y >> 8);
        for (int i = 0 ; i < spot_count; ++i, ++lid) {
            _lightingContribution.add_local_spot(scn_lights[LightIndex(lid)]);
        }

    #endif
#else // USE_PER_PIXEL_LIGHTING
        _lightingContribution.diffuse = in.diffuse;
    #ifdef USE_SPECULAR
        _lightingContribution.specular = in.specular;
    #endif
#endif // USE_PER_PIXEL_LIGHTING
    #ifdef AVOID_OVERLIGHTING
        _lightingContribution.diffuse = saturate(_lightingContribution.diffuse);
    #ifdef USE_SPECULAR
        _lightingContribution.specular = saturate(_lightingContribution.specular);
    #endif // USE_SPECULAR
    #endif // AVOID_OVERLIGHTING
#else // USE_LIGHTING
    _lightingContribution.diffuse = float3(1.);
#endif // USE_LIGHTING

    //
    // MARK: Populating the `_output` struct
    //
    
#ifdef USE_PBR
    { // combine IBL + lighting
        float3 albedo = _surface.diffuse.rgb;
        float3 diffuseAlbedo = mix(albedo, float3(0.0), _surface.metalness);

        // ambient
        float3 color = (_lightingContribution.ambient * _surface.ambientOcclusion) * albedo;
        color += _lightingContribution.diffuse * diffuseAlbedo;
#ifndef DISABLE_SPECULAR
        color += _lightingContribution.specular;
#endif
#ifdef USE_EMISSION
        color += _surface.emission.rgb;
#endif
#ifdef USE_MULTIPLY
        color *= _surface.multiply.rgb;
#endif
#ifdef USE_MODULATE
        color *= _lightingContribution.modulate;
#endif
        _output.color.rgb = color;
    }
#else // USE_PBR
    _output.color.rgb = illuminate(_surface, _lightingContribution);
#endif
    _output.color.a = _surface.diffuse.a;

#ifdef USE_FOG
    float fogFactor = pow(clamp(length(_surface.position.xyz) * scn_frame.fogParameters.x + scn_frame.fogParameters.y, 0., scn_frame.fogColor.a), scn_frame.fogParameters.z);
    _output.color.rgb = mix(_output.color.rgb, scn_frame.fogColor.rgb * _output.color.a, fogFactor);
#endif

#ifndef DIFFUSE_PREMULTIPLIED
    _output.color.rgb *= _surface.diffuse.a;
#endif
    
    //
    // MARK: Opacity and transparency
    //

#ifdef USE_TRANSPARENT // Either a map or a color
    
#ifdef USE_TRANSPARENCY
    _surface.transparent *= scn_commonprofile.transparency;
#endif
    
#ifdef USE_TRANSPARENCY_RGBZERO
#ifdef USE_NODE_OPACITY
    _output.color *= in.nodeOpacity;
#endif
    // compute luminance
    _surface.transparent.a = (_surface.transparent.r * 0.212671f) + (_surface.transparent.g * 0.715160f) + (_surface.transparent.b * 0.072169f);
    _output.color *= (float4(1.f) - _surface.transparent);
#else // ALPHA_ONE
#ifdef USE_NODE_OPACITY
    _output.color *= (in.nodeOpacity * _surface.transparent.a);
#else
    _output.color *= _surface.transparent.a;
#endif
#endif
#else
#ifdef USE_TRANSPARENCY // TRANSPARENCY without TRANSPARENT slot (nodeOpacity + diffuse.a)
#ifdef USE_NODE_OPACITY
    _output.color *= (in.nodeOpacity * scn_commonprofile.transparency);
#else
    _output.color *= scn_commonprofile.transparency;
#endif // NODE_OPACITY
#endif
#endif
    
    //
    // MARK: Fragment shader modifier
    //
    
#ifdef USE_FRAGMENT_MODIFIER
    // DoFragmentModifier START
    
    // DoFragmentModifier END
#endif
#if defined(USE_CLUSTERED_LIGHTING) && defined(DEBUG_CLUSTER_TILE)
    _output.color.rgb = mix(_output.color.rgb, float3(scn::debugColorForCount(clusterIndex.z).xyz), 0.1f);
    _output.color.rgb = mix(_output.color.rgb, float3(clusterIndex.x & 0x1 ^ clusterIndex.y & 0x1).xyz, 0.01f);
#endif
#ifdef DISABLE_LINEAR_RENDERING
    _output.color.rgb = scn::linear_to_srgb(_output.color.rgb);
#endif
    
#ifdef USE_DISCARD
    if (_output.color.a == 0.) // we could set a different limit here
        discard_fragment();
#endif

#ifdef USE_POINT_RENDERING
    if ((dfdx(pointCoord.x) < 0.5f) && (length_squared(pointCoord * 2.f - 1.f) > 1.f)) {
        discard_fragment();
    }
#endif
    
#ifdef USE_MOTIONBLUR

#ifdef USE_MULTIPLE_RENDERING
        _output.motionblur.xy = (half2(((in.fragmentPosition.xy-scn_frame_slice.viewport.zw)/scn_frame_slice.viewport.xy)*2.f-1.f) - half2((in.velocity.xy) / in.velocity.z)) * scn_frame_slice.motionBlurIntensity;
#else
    _output.motionblur.xy = (half2(in.fragmentPosition.xy*scn_frame.inverseResolution.xy*2.f-1.f) - half2((in.velocity.xy) / in.velocity.z)) * scn_frame.motionBlurIntensity;
#endif
    _output.motionblur.z = length(_output.motionblur.xy);
    _output.motionblur.w = half(-_surface.position.z);
#endif

#ifdef USE_NORMALS_OUTPUT
    _output.normals = half4( half3(_surface.normal.xyz), 0.h );
#endif
#ifdef USE_REFLECTANCE_ROUGHNESS_OUTPUT
#ifdef USE_PBR
    _output.reflectanceRoughnessOutput = half4( half3(_surface.diffuse.rgb * _lightingContribution.reflectance), half(_surface.roughness) );
#else // SSR on non pbr material is not supported
    _output.reflectanceRoughnessOutput = half4( 0.h );
#endif
#endif
    
#ifdef USE_OUTLINE
	_output.color.rgb = in.outlineHash;
#endif

    return _output;
}
