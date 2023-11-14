
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_utiltex_lib.glsl)
#pragma BLENDER_REQUIRE(raytrace_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)
#pragma BLENDER_REQUIRE(bsdf_sampling_lib.glsl)
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

void main()
{
  if (ssrDiffuseIntensity == 0.0) { //TODO check out of shader
    return;
  }
  vec4 rand = texelfetch_noise_tex(gl_FragCoord.xy);

  /* Decorrelate from AA. */
  vec2 random_px = floor(fract(rand.xy * 2.2074408460575947536) * 1.99999) - 0.8; //0.5
  rand.xy = fract(rand.xy * 3.671795724474602596); //3.2471795724474602596

  /* Randomly choose the pixel to start the ray from when tracing at lower resolution.
  * This method also make sure we always start from the center of a fullres texel. */
  vec2 uvs = (gl_FragCoord.xy + random_px * randomScale) / (targetSize * ssrUvScale);

  float depth = textureLod(maxzBuffer, uvs * hizUvScale.xy, 0.0).r;

  SsgiHitData data;
  data.ray_pdf_inv = 0.0;
  data.is_hit = false;
  data.hit_dir = vec3(0.0, 0.0, 0.0);
  /* Default: not hits. */
  ssgi_encode_hit_data(data, data.hit_dir, data.hit_dir, ssgiHitData, ssgiHitDepth);

  /* Early out - depth*/
  /* We can't do discard because we don't clear the render target. */
  if (depth == 1.0) {
    return;
  }

  /* Using view space */
  vec3 vP = get_view_space_from_depth(uvs, depth);
  vec3 P = transform_point(ViewMatrixInverse, vP);
  vec3 vV = viewCameraVec(vP);
  vec3 V = cameraVec(P);
  vec3 vN = normal_decode(texture(normalBuffer, uvs, 0).rg, vV);
  vec3 N = transform_direction(ViewMatrixInverse, vN);

  /* Retrieve pixel data */
  vec4 difcol_roughness = texture(ssgiInputBuffer, uvs, 0).rgba;

  /* Early out - bsdf */
  if (dot(difcol_roughness.rgb, vec3(1.0)) == 0.0) {
    return;
  }

  /* Importance sampling bias */
  //rand.x = mix(rand.x, 0.0, ssrDiffuseDebug_h);

  vec3 vT, vB;
  make_orthonormal_basis(vN, vT, vB); /* Generate tangent space */





  float pdf;
  /* new GGX sampling */
  vec3 vH = sample_ggx(rand.xzw, 1.0, vV, vN, vT, vB, pdf); //TODO cosine //arg 2 - alpha
  vec3 vR = reflect(-vV, vH);


  if (isnan(pdf)) {
    /* Seems that somethings went wrong.
     * This only happens on extreme cases where the normal deformed too much to have any valid
     * reflections. */
    return;
  }

  Ray ray;
  ray.origin = vP;
  ray.direction = vR * 1e16;

  RayTraceParameters params;
  params.thickness = ssrDiffuseThickness;
  params.jitter = rand.y;
  params.trace_quality = ssrDiffuseQuality;
  params.roughness = 1.0;

  vec3 hit_sP;
  data.is_hit = raytrace(ray, params, true, false, hit_sP); //Add separate raytrace func?
  data.ray_pdf_inv = safe_rcp(pdf);

  ssgi_encode_hit_data(data, hit_sP, ray.origin, ssgiHitData, ssgiHitDepth);
}
