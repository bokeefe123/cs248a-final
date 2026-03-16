implementing "../renderer.slang";

import light;
import portal;

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<Portal.PortalHitResult> rayPortalIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<Portal.PortalHitResult> closestPortalHit = Optional<Portal.PortalHitResult>();
    for (uint i = 0; i < uniforms.portalCount; i++)
    {
        Optional<Portal.PortalHitResult> portalHit = uniforms.portalBuf[i].hit(ray, i);
        if (portalHit.hasValue)
        {
            if (!closestPortalHit.hasValue || portalHit.value.t < closestPortalHit.value.t)
            {
                closestPortalHit = portalHit;
            }
        }
    }
    return closestPortalHit;
}

public Optional<RectangularLight.LightHitResult> rayRectangularLightIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RectangularLight.LightHitResult> closestLightHitResult = Optional<RectangularLight.LightHitResult>();
    for (int i = 0; i < uniforms.rectangularLightCount; i++)
    {
        Optional<RectangularLight.LightHitResult> lightHitResult = uniforms.rectangularLightBuf[i].hit(ray);
        if (lightHitResult.hasValue)
        {
            if (!closestLightHitResult.hasValue || lightHitResult.value.t < closestLightHitResult.value.t)
            {
                closestLightHitResult = lightHitResult;
            }
        }
    }
    return closestLightHitResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms, bool occlusionTest = false)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();

    // IMPORTANT NOTE: Check the modified signature of `rayMeshIntersection` function in ray_mesh_intersection.slang.
    // Can you modify your existing code to make use of this flag? This is helpful in `isLightOccluded` function.

    // TODO: Student implementation starts here.
    // if (occlusionTest) {
    //     return allResult;
    // }

    Optional<RayHitResult> closestHitResult;
    Optional<RayHitResult> hitResult;
    if (uniforms.useBVH) {
        Optional<BVH<Triangle>.HitResult> bvhHitResult = uniforms.bvh.hit(ray);
        if (bvhHitResult.hasValue) {
            closestHitResult = Optional<RayHitResult>(bvhHitResult.value.rayHitResult);
            hitTriangle = Optional<Triangle>(bvhHitResult.value.primitive);
        }
    } else {
        for (uint i = 0; i < uniforms.triangleCount; i++)
        {
            Triangle triangle = uniforms.triangleBuf[i];
            hitResult = triangle.hit(ray);
            if (!hitResult.hasValue) {}
            else if (!closestHitResult.hasValue) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }
            else if (hitResult.value.t < closestHitResult.value.t) {
                closestHitResult = hitResult;
                hitTriangle = Optional<Triangle>(triangle);
            }  
        }
    }
    result = closestHitResult;

    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}
