implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

// Helper function to get texture UV at an offset screen position
// Returns an Optional - empty if the offset ray doesn't hit the triangle
Optional<float2> getTexUVAtOffset(RendererUniform uniforms, Triangle triangle, float2 offsetUV)
{
    Ray ray = uniforms.camera.generateRay(offsetUV);
    Optional<RayHitResult> result = triangle.hit(ray);
    if (!result.hasValue) {
        return Optional<float2>();
    }
    float3 hitPoint = ray.origin + result.value.t * ray.direction;
    float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
    return Optional<float2>(triangle.getUV(barycentricCoord));
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    
    // One pixel step in normalized screen coordinates
    float dx = 1.0 / float(uniforms.camera.canvasSize.x);
    float dy = 1.0 / float(uniforms.camera.canvasSize.y);

    // Get texture UVs at screen positions offset by one pixel in x and y
    // Optional<float2> texUV_opt = getTexUVAtOffset(uniforms, triangle, uv);
    // if (!texUV_opt.hasValue) {
    //     return 0.0;
    // }
    Optional<float2> texUV_pos_dx_opt = getTexUVAtOffset(uniforms, triangle, uv + float2(dx, 0.0));
    Optional<float2> texUV_neg_dx_opt = getTexUVAtOffset(uniforms, triangle, uv - float2(dx, 0.0));
    Optional<float2> texUV_pos_dy_opt = getTexUVAtOffset(uniforms, triangle, uv + float2(0.0, dy));
    Optional<float2> texUV_neg_dy_opt = getTexUVAtOffset(uniforms, triangle, uv - float2(0.0, dy));

    // If offset rays miss the triangle, use the current tex_uv as fallback
   
    float2 texUV_dx = texUV_pos_dx_opt.hasValue ? texUV_pos_dx_opt.value : texUV_neg_dx_opt.hasValue ? texUV_neg_dx_opt.value : tex_uv;
    float2 texUV_dy = texUV_pos_dy_opt.hasValue ? texUV_pos_dy_opt.value : texUV_neg_dy_opt.hasValue ? texUV_neg_dy_opt.value : tex_uv;

    float2 duvdx = (texUV_dx - tex_uv) / dx;
    float2 duvdy = (texUV_dy - tex_uv) / dy;

    float rho_x = length(duvdx);
    float rho_y = length(duvdy);

    float rho = max(rho_x, rho_y);
    level = log2(max(rho, 1.0));

    return level;

    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
