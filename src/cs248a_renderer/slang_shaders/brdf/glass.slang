implementing "../brdf.slang";

public struct GlassBRDF
{
    public float3 albedo;
    public float ior;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {

        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);
        
        // TODO: Student implementation starts here.
        float cosTheta = dot(rayOut, normal);
        float3 n = normal;
        float eta;

        if (cosTheta > 0.0) {
            // Entering
            eta = 1.0 / this.ior;
        } else {
            // Exiting
            eta = this.ior;
            n = -normal;
            cosTheta = -cosTheta;
        }

        float discriminant = 1.0 - (eta * eta) * (1.0 - cosTheta * cosTheta);

        if (discriminant < 0.0) {
            // Total internal reflection — reflect like a mirror
            sampleDirection = -rayOut + 2.0 * dot(rayOut, n) * n;
        } else {
            // Refract using Snell's law
            float3 rayIn = -eta * rayOut + (eta * cosTheta - sqrt(discriminant)) * n;
            sampleDirection = normalize(rayIn);
        }
        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        // For dielectric materials, the PDF is 1.0 for the single direction
        // (either refracted or reflected due to total internal reflection)
        return 1.0;
    }
}
