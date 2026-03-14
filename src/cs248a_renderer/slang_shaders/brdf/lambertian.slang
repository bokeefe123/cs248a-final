implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.

            float theta = acos(sqrt(randomEpsilon.y));
            float phi = 2.0 * PI * randomEpsilon.x;
            float3 localSampleDirection = float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
            sampleDirection = mul(localToWorldMatrix, localSampleDirection);
            sampleDirection = normalize(sampleDirection);

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            float phi = 2.0 * PI * randomEpsilon.x;
            float cosTheta = randomEpsilon.y;
            float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            // Local space: y is "up" (the normal direction)
            float3 localSampleDirection = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
            // Convert to world space
            sampleDirection = mul(localToWorldMatrix, localSampleDirection);
            sampleDirection = normalize(sampleDirection);

            // TODO: Student implementation ends here.
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.
            
            pdf = abs(dot(rayIn, normal)) / PI;
           

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.
            pdf = 1.0 / (2.0 * PI);
            // TODO: Student implementation ends here.
        }
        
        return pdf;
    }
}
