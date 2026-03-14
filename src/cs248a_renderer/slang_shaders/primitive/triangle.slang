implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        float3 pMin = min(min(vertices[0], vertices[1]), vertices[2]);
        float3 pMax = max(max(vertices[0], vertices[1]), vertices[2]);
        return BoundingBox(pMin, pMax);
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;

        // TODO: Student implementation starts here.
        float3 normalVector = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
        float triangleArea = length(normalVector);
        float3 unitNormal = normalVector / triangleArea;
        
        float alpha = dot(cross(p - vertices[1], p - vertices[2]), unitNormal) / triangleArea;
        float beta = dot(cross(p - vertices[2], p - vertices[0]), unitNormal) / triangleArea;
        float gamma = dot(cross(p - vertices[0], p - vertices[1]), unitNormal) / triangleArea;

        // TODO: Student implementation ends here.

        return float3(alpha, beta, gamma);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.

        float3 normalVector = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
        float triangleArea = length(normalVector);
        // normalVector = normalize(normalVector);
        float offset = dot(normalVector, vertices[0]);
        float t = (offset - dot(normalVector, ray.origin)) / dot(normalVector, ray.direction);
        float3 p = ray.origin + t * ray.direction;

        float alpha = length(cross(p - vertices[1], p - vertices[2])) / (triangleArea);
        float beta = length(cross(p - vertices[2], p - vertices[0])) / (triangleArea);
        float gamma = length(cross(p - vertices[0], p - vertices[1])) / (triangleArea);
        if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1 && abs(alpha + beta + gamma - 1.0) < 1e-4) {
            if (t < ray.tRange.x || t > ray.tRange.y) {
                return Optional<RayHitResult>();
            }
            return Optional<RayHitResult>(RayHitResult(t, normalize(normalVector)));
        } 

        return Optional<RayHitResult>();

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
