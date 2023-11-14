
#pragma BLENDER_REQUIRE(common_math_lib.glsl)
#pragma BLENDER_REQUIRE(common_math_geom_lib.glsl)
#pragma BLENDER_REQUIRE(common_colorpacking_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_glossy_lib.glsl)
#pragma BLENDER_REQUIRE(closure_eval_lib.glsl)
#pragma BLENDER_REQUIRE(lightprobe_lib.glsl)
#pragma BLENDER_REQUIRE(bsdf_common_lib.glsl)
#pragma BLENDER_REQUIRE(surface_lib.glsl)
#pragma BLENDER_REQUIRE(effect_reflection_lib.glsl)

/* 
https://jo.dreggn.org/home/2010_atrous.pdf 
https://www.shadertoy.com/view/ldKBzG
*/

#define KERNEL_SIZE 25
uniform float kernel[KERNEL_SIZE];
uniform vec2 offset[KERNEL_SIZE];

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec3 doAtrous (in vec2 uv, sampler2D source)
{
  vec2 offset[25];
  offset[0] = vec2(-2,-2);
  offset[1] = vec2(-1,-2);
  offset[2] = vec2(0,-2);
  offset[3] = vec2(1,-2);
  offset[4] = vec2(2,-2);
  
  offset[5] = vec2(-2,-1);
  offset[6] = vec2(-1,-1);
  offset[7] = vec2(0,-1);
  offset[8] = vec2(1,-1);
  offset[9] = vec2(2,-1);
  
  offset[10] = vec2(-2,0);
  offset[11] = vec2(-1,0);
  offset[12] = vec2(0,0);
  offset[13] = vec2(1,0);
  offset[14] = vec2(2,0);
  
  offset[15] = vec2(-2,1);
  offset[16] = vec2(-1,1);
  offset[17] = vec2(0,1);
  offset[18] = vec2(1,1);
  offset[19] = vec2(2,1);
  
  offset[20] = vec2(-2,2);
  offset[21] = vec2(-1,2);
  offset[22] = vec2(0,2);
  offset[23] = vec2(1,2);
  offset[24] = vec2(2,2);

  float kernel[25];
  kernel[0] = 0.00390625f;
  kernel[1] = 0.015625f;
  kernel[2] = 0.0234375f;
  kernel[3] = 0.015625f;
  kernel[4] = 0.00390625f;
  
  kernel[5] = 0.015625f;
  kernel[6] = 0.0625f;
  kernel[7] = 0.09375f;
  kernel[8] = 0.0625f;
  kernel[9] = 0.015625f;
  
  kernel[10] = 0.0234375f;
  kernel[11] = 0.09375f;
  kernel[12] = 0.140625f;
  kernel[13] = 0.09375f;
  kernel[14] = 0.0234375f;
  
  kernel[15] = 0.015625f;
  kernel[16] = 0.0625f;
  kernel[17] = 0.09375f;
  kernel[18] = 0.0625f;
  kernel[19] = 0.015625f;
  
  kernel[20] = 0.00390625f;
  kernel[21] = 0.015625f;
  kernel[22] = 0.0234375f;
  kernel[23] = 0.015625f;
  kernel[24] = 0.00390625f;

  /* input block */ //Could move some to uniforms
  float sizevariance = map(samplePoolOffset, 0.0, 32.0, 0.0, 1.0);  //rename samplePoolOffset to more appropriate in this case
  //float sizevarianceinv = map(samplePoolOffset, 0.0, 256.0, 32.0, 1.0);
  float c_phi = 1.0;
  float n_phi = mix(1.0, 0.0002, ssrDiffuseFnweight);//  / sizevarianceinv;
  float p_phi = mix(1.0, 0.0001, ssrDiffuseFdweight);// / sizevarianceinv;
  float stepwidth = ssrDiffuseFsize / mix(1.0, 5.0, sizevariance);
  /* input block end */

  vec3 sum = vec3(0.0);
  vec2 step = vec2(1/targetSize.x, 1/targetSize.y); // resolution

  vec4 cval = texture(source, uv, 0).rgba;
  vec4 nval = texture(normalBuffer, uv, 0).rgba;
  vec4 pval = vec4(texture(maxzBuffer, (uv * hizUvScale.xy), 0).r);
  pval = pval * pval; // Depth Squared
  //AO weight
  float aoweight = 0.01 * ssrDiffuseFaoweight;
  vec4 difcol_roughness  = vec4(texture(ssgiInputBuffer, uv, 0).a); //TODO Clean Up
  vec4 aoval = vec4(difcol_roughness.a) * aoweight;

  float cum_w = 0.0;
    for(int i = 0; i < KERNEL_SIZE; i++) {

    vec2 uv_s = uv + offset[i]*step*stepwidth;

    vec4 ctmp = texture(source, uv_s, 0).rgba;
    vec4 t = cval - ctmp;
    float dist2 = dot(t,t);
    float c_w = min(exp(-(dist2)/c_phi), 1.0);

    vec4 ntmp = texture(normalBuffer, uv_s, 0).rgba; //Norm - vec3
    t = nval - ntmp;
    dist2 = max(dot(t,t)/(stepwidth*stepwidth),0.0); //?? //dist2 = max(dot(t,t), 0.0); 
    float n_w = min(exp(-(dist2)/n_phi), 1.0);

    vec4 ptmp =  vec4(texture(maxzBuffer, (uv_s * hizUvScale.xy), 0).r);
    ptmp = ptmp * ptmp; // Depth Squared
    t = pval - ptmp;
    dist2 = dot(t,t);
    float p_w = min(exp(-(dist2)/p_phi),1.0);

    //AO Weight
    difcol_roughness  = vec4(texture(ssgiInputBuffer, uv_s, 0).a); //TODO Clean Up
    vec4 aotmp = vec4(difcol_roughness.a) * aoweight;
    t = aoval - aotmp;
    dist2 = dot(t,t);
    float ao_w = min(exp(-(dist2)/p_phi),1.0);

    float weight = c_w * n_w * p_w * ao_w * kernel[i];

    sum += ctmp.rgb * weight;
    cum_w += weight;
  }
  return mix((sum/cum_w).rgb, cval.rgb, 0.2);
}

 void main()
 {
  float depth = textureLod(maxzBuffer, uvcoordsvar.xy * hizUvScale.xy, 0.0).r;

  if (depth == 1.0) {
    discard;
  }

  ivec2 texel = ivec2(gl_FragCoord.xy);
  vec4 difcol_roughness  = texelFetch(ssgiInputBuffer, texel, 0).rgba;

  if (max_v3(difcol_roughness.rgb) <= 0.0) {
    discard;
  }

  //vec2 texture_size = vec2(textureSize(ssgiHitBuffer, 0)); // TODO fix half res trace output
  vec3 prefilter = texture(ssgiFilterInput, uvcoordsvar.xy).rgb;

  /* Do toggle out of shader */
  if (ssrDiffuseFilter > 0.0 && samplePoolOffset > ssrDiffuseFsamples) {
    vec4 filtered = vec4(0.0);
    filtered = vec4(doAtrous(uvcoordsvar.xy, ssgiFilterSecInput), 1.0);

    /* gi * albedo * AO * intensity */

    fragColor = vec4(
      mix(prefilter.rgb, filtered.rgb, ssrDiffuseFilter) //GI output raw/denoised mix
      * 
      difcol_roughness.rgb //albedo
      *
      mix(1.0, map(difcol_roughness.a, 0.0, 1.0, ssrDiffuseAoLimit, 1.0), ssrDiffuseAo) //AO
      *
      ssrDiffuseIntensity //Master intensity
      ,1.0);

  }
  else {
    fragColor = vec4(
      prefilter.rgb //Unfiltered GI
      *
      difcol_roughness.rgb //albedo
      *
      mix(1.0, map(difcol_roughness.a, 0.0, 1.0, ssrDiffuseAoLimit, 1.0), ssrDiffuseAo) //AO
      *
      ssrDiffuseIntensity //Master intensity
      , 1.0);
  }
 }
