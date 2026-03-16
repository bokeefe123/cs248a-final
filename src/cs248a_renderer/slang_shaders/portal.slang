module portal;

import math;
import primitive;

public struct Portal
{
    public float3 bottomLeftVertex;
    public float3 bottomEdge;
    public float3 leftEdge;
    public float3 normal;
    public uint partnerIndex;
    public float3 glowColor;
    public float glowWidth;
    public float glowIntensity;

    public struct PortalHitResult
    {
        public float t;
        public float3 normal;
        public float2 localUV;
        public uint portalIndex;
    }

    public float getGlowFactor(float2 localUV)
    {
        float edgeDist = min(min(localUV.x, 1.0 - localUV.x), min(localUV.y, 1.0 - localUV.y));
        if (edgeDist >= this.glowWidth)
            return 0.0;
        float t = 1.0 - edgeDist / this.glowWidth;
        return t * t;
    }

    public Optional<PortalHitResult> hit(Ray ray, uint myIndex)
    {
        float3 v0 = this.bottomLeftVertex;
        float3 v1 = v0 + this.bottomEdge;
        float3 v2 = v1 + this.leftEdge;
        float3 v3 = v0 + this.leftEdge;

        float3[3] vertices1 = float3[3](v0, v1, v2);
        float3[3] vertices2 = float3[3](v0, v2, v3);
        float3[3] normals = float3[3](this.normal, this.normal, this.normal);
        float2[3] uvs = float2[3](float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0));
        float3[3] colors = float3[3](float3(0.0), float3(0.0), float3(0.0));

        Triangle triangle1 = Triangle(vertices1, normals, uvs, colors);
        Triangle triangle2 = Triangle(vertices2, normals, uvs, colors);

        Optional<RayHitResult> hitResult = Optional<RayHitResult>();
        Optional<RayHitResult> hitResult1 = triangle1.hit(ray);

        if (hitResult1.hasValue && hitResult1.value.t <= ray.tRange.y)
        {
            hitResult = hitResult1;
        }
        else
        {
            Optional<RayHitResult> hitResult2 = triangle2.hit(ray);
            if (hitResult2.hasValue && hitResult2.value.t <= ray.tRange.y)
            {
                hitResult = hitResult2;
            }
        }

        if (hitResult.hasValue)
        {
            float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
            float2 localUV = computeLocalUV(hitPoint);
            return Optional<PortalHitResult>(PortalHitResult(hitResult.value.t, hitResult.value.normal, localUV, myIndex));
        }

        return Optional<PortalHitResult>();
    }

    float2 computeLocalUV(float3 hitPoint)
    {
        float3 offset = hitPoint - this.bottomLeftVertex;
        float u = dot(offset, normalize(this.bottomEdge)) / length(this.bottomEdge);
        float v = dot(offset, normalize(this.leftEdge)) / length(this.leftEdge);
        return float2(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
    }

    public Ray getExitRay(float2 localUV, float3 incomingDir, Portal partner)
    {
        const float EPSILON = 1e-2;

        float3 exitPoint = partner.bottomLeftVertex + partner.bottomEdge * (1.0 - localUV.x) + partner.leftEdge * localUV.y;

        float3 entryRight = normalize(this.bottomEdge);
        float3 entryUp = normalize(this.leftEdge);
        float3 entryNormal = this.normal;

        float3 exitRight = normalize(partner.bottomEdge);
        float3 exitUp = normalize(partner.leftEdge);
        float3 exitNormal = partner.normal;

        float dRight = dot(incomingDir, entryRight);
        float dUp = dot(incomingDir, entryUp);
        float dNormal = dot(incomingDir, entryNormal);

        // 180° rotation around the up axis: negate right and normal components.
        // Entering a portal face-on means exiting the partner facing outward,
        // which flips both the lateral and depth directions.
        float3 exitDir = normalize(-dRight * exitRight + dUp * exitUp - dNormal * exitNormal);

        float3 origin = exitPoint + EPSILON * exitNormal;

        return Ray(origin, exitDir, float2(0.0, float.maxValue));
    }
}
