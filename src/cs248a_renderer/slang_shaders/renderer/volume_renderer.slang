implementing "../renderer.slang";

import math;
import texture;

public static const uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    
    // Generate camera ray in world space
    Ray ray = uniforms.camera.generateRay(uv);
    ray.tRange.y = min(ray.tRange.y, tMax);
    
    Optional<RayHitResult> closestHitResult;
    for (uint i = 0; i < uniforms.surfaceVolumeCount; i++)
    {
        Volume volume = uniforms.surfaceVolumeBuf[i];
        Optional<RayHitResult> hitResult = volume.hit(ray, uniforms.surfaceVolumeTexBuf);
        if (!hitResult.hasValue) {}
        else if (!closestHitResult.hasValue) {
            closestHitResult = hitResult;
        }
        else if (hitResult.value.t < closestHitResult.value.t) {
            closestHitResult = hitResult;
        }  
    }
    
    if (closestHitResult.hasValue) {
        ray.tRange.y = min(ray.tRange.y, closestHitResult.value.t);
        primColor = float4(closestHitResult.value.normal * 0.5 + 0.5, 1.0);
    }
    
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);
    
    float4 color = accumulateColor<4>(
        localRay,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        uniforms.volume.tex
    );
    
    return color;
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    
    Optional<RayHitResult> closestHitResult;
    for (uint i = 0; i < uniforms.surfaceVolumeCount; i++)
    {
        Volume volume = uniforms.surfaceVolumeBuf[i];
        Optional<RayHitResult> hitResult = volume.hit(ray, uniforms.surfaceVolumeTexBuf);
        if (!hitResult.hasValue) {}
        else if (!closestHitResult.hasValue) {
            closestHitResult = hitResult;
        }
        else if (hitResult.value.t < closestHitResult.value.t) {
            closestHitResult = hitResult;
        }  
    }
    
    if (closestHitResult.hasValue) {
        ray.tRange.y = min(ray.tRange.y, closestHitResult.value.t);
    }  
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);

    DifferentialPtrPair<DiffTexture3D<float, 4>> diffAccumulateColorPair = 
        DifferentialPtrPair<DiffTexture3D<float, 4>>(
            uniforms.volume.tex,
            uniforms.volume.dTex
        );
    float4 backgroundColor = uniforms.ambientColor;
    bwd_diff(accumulateColor<4>)(localRay, uniforms.volume.bound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, backgroundColor, diffAccumulateColorPair, outGrad);
    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    
    float2 tHit = volumeBound.hit(ray);
    if (tHit.x < 0.0 || tHit.y < 0.0)
    {
        return backgroundColor;
    }
    float tEnter = max(tHit.x, ray.tRange.x);
    float tExit = min(tHit.y, ray.tRange.y);
    if (tEnter >= tExit) {
        return backgroundColor;
    }

    float3 accumulatedColor = float3(0.0, 0.0, 0.0);
    float transmittance = 1.0;
    float3 volumeMin = volumeBound.pMin;
    float3 volumeSize = volumeBound.pMax - volumeBound.pMin;

    [MaxIters(MAX_STEPS)]
    for (uint step = 0; step < maxSteps; step++)
    {
        float t = tEnter + float(step) * stepSize;
        if (t >= tExit)
        {
            break;
        }
        if (transmittance < EPSILON)
        {
            break;
        }
        
        float3 pos = ray.origin + t * ray.direction;
        
        float3 uvw = (pos - volumeMin) / volumeSize;
        
        float4 sample = sampleTrilinear(data, uvw);
        float3 emission = sample.rgb;
        float density = sample.a * densityScale;
        
        float alpha = 1.0 - exp(-density * stepSize);
        
        accumulatedColor += transmittance * alpha * emission;
        
        transmittance *= (1.0 - alpha);
    }
    
    float3 finalColor = accumulatedColor + transmittance * backgroundColor.rgb;
    float finalAlpha = (1.0 - transmittance) + transmittance * backgroundColor.a;
    
    return float4(finalColor, finalAlpha);

    // TODO: Student implementation ends here.
}
