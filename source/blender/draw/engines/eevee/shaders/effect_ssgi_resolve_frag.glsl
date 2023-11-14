
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_glossy_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)
#pragma BLENDER_REQUIRE(bsdf_common_lib.glsl)
#pragma BLENDER_REQUIRE(surface_lib.glsl)
#pragma BLENDER_REQUIRE(effect_reflection_lib.glsl)
#pragma BLENDER_REQUIRE(common_colorpacking_lib.glsl)

/* Based on:
 * "Stochastic Screen Space Reflections"
 * by Tomasz Stachowiak.
 * https://www.ea.com/frostbite/news/stochastic-screen-space-reflections
 * and
 * "Stochastic all the things: raytracing in hybrid real-time rendering"
 * by Tomasz Stachowiak.
 * https://media.contentapi.ea.com/content/dam/ea/seed/presentations/dd18-seed-raytracing-in-hybrid-real-time-rendering.pdf
 * and
 * adapted to do diffuse reflections only, based on the default Eevee SSR implementation.
 */

vec4 ssgi_get_scene_color_and_mask(vec3 hit_vP, float mip)
{
  vec2 uv;
  /* Find hit position in previous frame. */
  /* TODO Combine matrices. */
  vec3 hit_P = transform_point(ViewMatrixInverse, hit_vP);
  /* TODO real reprojection with motion vectors, etc... */
  uv = project_point(pastViewProjectionMatrix, hit_P).xy * 0.5 + 0.5;

  vec3 color;
  color = textureLod(colorBuffer, uv * hizUvScale.xy, mip).rgb;

  /* Clamped brightness. */
  float luma = max_v3(color);
  color *= 1.0 - max(0.0, luma - ssrDiffuseClamp) * safe_rcp(luma);

  /* Edge fade mask */
  //float mask = screen_border_mask(uv); //Temporarily disaled

  //return vec4(color, mask);
  return vec4(color, 0.0);
}

void resolve_reflection_sample(vec2 sample_uv,
                               vec3 vP,
                               vec3 vN,
                               vec3 vV,
                               vec3 V,
                               vec3 P,
                               vec3 N,
                               inout float weight_accum,
                               inout vec4 ssgi_accum)
{
  vec4 hit_data = texture(ssgiHitBuffer, sample_uv);
  float hit_depth = texture(ssgiHitDepth, sample_uv).r;
  SsgiHitData data = ssgi_decode_hit_data(hit_data, hit_depth);

  float hit_dist = length(data.hit_dir);

  /* Slide 54. */
  // Ray hit weight
  float bsdf = bsdf_ggx(vN, data.hit_dir / hit_dist, vV, 1.0);

  float weight = bsdf * data.ray_pdf_inv;

  /* Do not add light if ray has failed but still weight it. */ //Using to inject probe lighting
   if (!data.is_hit) {
    weight_accum += weight;

    vec3 inject_normal = transform_direction(ViewMatrixInverse, data.hit_dir);


    /*using fallback cubemap*/

    //float depth = textureLod(maxzBuffer, uvcoordsvar.xy * hizUvScale.xy, 0.0).r;
    //viewPosition = get_view_space_from_depth(uvcoordsvar.xy, depth);
    //worldPosition = transform_point(ViewMatrixInverse, viewPosition);

    vec4 cubemap_sample_probe = vec4(0.0);
    if (ssrDiffuseProbeTrace > 0) {
      fallback_cubemap_ssgi(inject_normal, P, cubemap_sample_probe);
    }
    vec3 cubemap_sample = cubemap_sample_probe.rgb;
    /*clamping for noise reduction, intensity*/
    cubemap_sample = clamp(cubemap_sample, vec3(0.0), vec3(ssrDiffuseProbeClamp));
    cubemap_sample *= ssrDiffuseProbeIntensity;

    // Ray Tracing Gems Chapter 25
    // vec3 fireflyRejectionVariance(vec3 radiance, vec3 variance, vec3 shortMean, vec3 dev) //TODO //Consider at denoise
    // {
    //     vec3 dev = sqrt(max(1.0e-5, variance));
    //     vec3 highThreshold = 0.1 + shortMean + dev * 8.0;
    //     vec3 overflow = max(0.0, radiance - highThreshold);
    //     return radiance - overflow;
    // }

    /*end*/



    //ssgi_accum += vec4(cubemap_sample, 0.0) * ssrThickness * weight; //TODO!!! double check thickness
    ssgi_accum += vec4(cubemap_sample, 0.0) * weight; 
    return;

  }


  vec3 hit_vP = vP + data.hit_dir;

  /* Precalculated */
  //float cone_cos = 11.0352189249; //Unused
  float cone_tan = 0.9958856386;

  /* Compute cone footprint in screen space. */
  float cone_footprint = hit_dist * cone_tan;
  float homcoord = ProjectionMatrix[2][3] * hit_vP.z + ProjectionMatrix[3][3];
  cone_footprint *= max(ProjectionMatrix[0][0], ProjectionMatrix[1][1]) / homcoord;
  cone_footprint *= ssrDiffuseResolveBias * 0.5;
  /* Estimate a cone footprint to sample a corresponding mipmap level. */
  float mip = log2(cone_footprint * max_v2(vec2(textureSize(ssgiInputBuffer, 0))));

  vec4 radiance_mask = ssgi_get_scene_color_and_mask(hit_vP, mip);

  ssgi_accum += radiance_mask * weight;
  weight_accum += weight;
}

/* NOTE(Metal): For Apple silicon GPUs executing this particular shader, by default, memory read
 * pressure is high while ALU remains low. Packing the sample data into a smaller format balances
 * this trade-off by reducing local shader register pressure and expensive memory look-ups into
 * spilled local shader memory, resulting in an increase in performance of 20% for this shader. */
#ifdef GPU_METAL
#  define SAMPLE_STORAGE_TYPE uchar
#  define pack_sample(x, y) uchar(((uchar(x + 2)) << uchar(3)) + (uchar(y + 2)))
#  define unpack_sample(x) vec2((char(x) >> 3) - 2, (char(x) & 7) - 2)
#else
#  define SAMPLE_STORAGE_TYPE vec2
#  define pack_sample(x, y) SAMPLE_STORAGE_TYPE(x, y)
#  define unpack_sample(x) x
#endif

vec4 raytrace_resolve_diffuse(vec4 difcol_roughness, vec3 viewPosition, vec3 worldPosition, vec3 worldNormal, out vec4 ssgi_radiance)
{
/* Note: Reflection samples declared in function scope to avoid per-thread memory pressure on
* tile-based GPUs e.g. Apple Silicon. */
const SAMPLE_STORAGE_TYPE resolve_sample_offsets[36] = SAMPLE_STORAGE_TYPE[36](
    /* Set 1. */
    /* First Ring (2x2). */
    pack_sample(0, 0),
    /* Second Ring (6x6). */
    pack_sample(-1, 3),
    pack_sample(1, 3),
    pack_sample(-1, 1),
    pack_sample(3, 1),
    pack_sample(-2, 0),
    pack_sample(3, 0),
    pack_sample(2, -1),
    pack_sample(1, -2),
    /* Set 2. */
    /* First Ring (2x2). */
    pack_sample(1, 1),
    /* Second Ring (6x6). */
    pack_sample(-2, 3),
    pack_sample(3, 3),
    pack_sample(0, 2),
    pack_sample(2, 2),
    pack_sample(-2, -1),
    pack_sample(1, -1),
    pack_sample(0, -2),
    pack_sample(3, -2),
    /* Set 3. */
    /* First Ring (2x2). */
    pack_sample(0, 1),
    /* Second Ring (6x6). */
    pack_sample(0, 3),
    pack_sample(3, 2),
    pack_sample(-2, 1),
    pack_sample(2, 1),
    pack_sample(-1, 0),
    pack_sample(-2, -2),
    pack_sample(0, -1),
    pack_sample(2, -2),
    /* Set 4. */
    /* First Ring (2x2). */
    pack_sample(1, 0),
    /* Second Ring (6x6). */
    pack_sample(2, 3),
    pack_sample(-2, 2),
    pack_sample(-1, 2),
    pack_sample(1, 2),
    pack_sample(2, 0),
    pack_sample(-1, -1),
    pack_sample(3, -1),
    pack_sample(-1, -2));
  /*metal fix end*/

  vec4 ssgi_accum = vec4(0.0);
  float weight_acc = 0.0;

  vec3 V, P, N;

  V = viewPosition;
  P = worldPosition;
  N = worldNormal;

  /* Using view space */
  vec3 vV = transform_direction(ViewMatrix, V);
  vec3 vP = transform_point(ViewMatrix, P);
  vec3 vN = transform_direction(ViewMatrix, N);

  int sample_pool = int((uint(gl_FragCoord.x) & 1u) + (uint(gl_FragCoord.y) & 1u) * 2u);
  sample_pool = (sample_pool + (samplePoolOffset / 5)) % 4;

  for (int i = 0; i < resolve_samples_count; i++) {
    int sample_id = sample_pool * resolve_samples_count + i;
    vec2 texture_size = vec2(textureSize(ssgiHitBuffer, 0));
    vec2 sample_texel = texture_size * uvcoordsvar.xy * ssrUvScale;
    vec2 sample_uv = (sample_texel + unpack_sample(resolve_sample_offsets[sample_id])) / texture_size;

    resolve_reflection_sample(
        sample_uv, vP, vN, vV, V, P, N, weight_acc, ssgi_accum);
  }

  /* Compute SSGI contribution */
  ssgi_accum *= safe_rcp(weight_acc);

  /* multiply with mask */
  ssgi_radiance.rgb = ssgi_accum.rgb * ssgi_accum.a;
  ssgi_radiance.rgb = ssgi_accum.rgb;
  //ssgi_accum -= ssgi_accum.a; //Leaving in for potentintial future use

  ssgi_radiance = vec4(ssgi_radiance.rgb, 1.0);
  return ssgi_radiance;
}

void main()
{
  if (ssrDiffuseIntensity == 0.0) { //Do check elsewhere
    discard;
  }

  float depth = textureLod(maxzBuffer, uvcoordsvar.xy * hizUvScale.xy, 0.0).r;

  if (depth == 1.0) {
    discard;
  }
  //TODO add intel & metal exceptions

  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 difcol_roughness = texelFetch(ssgiInputBuffer, texel, 0).rgba;

  vec3 brdf = difcol_roughness.rgb * ssrDiffuseIntensity;

  if (max_v3(brdf) <= 0.0) {
    discard;
  }

  FragDepth = depth;

  viewPosition = get_view_space_from_depth(uvcoordsvar.xy, depth);
  worldPosition = transform_point(ViewMatrixInverse, viewPosition);

  vec2 normal_encoded = texelFetch(normalBuffer, texel, 0).rg;
  viewNormal = normal_decode(normal_encoded, viewCameraVec(viewPosition));
  worldNormal = transform_direction(ViewMatrixInverse, viewNormal);

  vec4 ssgi_radiance = vec4(0.0);
  ssgi_radiance = raytrace_resolve_diffuse(difcol_roughness, viewPosition, worldPosition, worldNormal, ssgi_radiance);

  //fragColor = vec4(ssgi_radiance.rgb * brdf, 1.0); //Unused going to filter stage instead
  ssgiFilterInput = vec4(ssgi_radiance.rgb, 1.0); //Output to filter step
}
