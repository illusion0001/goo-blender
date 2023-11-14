/* don't do this here - should be in lib */
float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

void node_ambient_occlusion(vec4 color,
                            float dist,
                            vec3 normal,
                            float softness,
                            const float inverted,
                            const float sample_count,
                            out vec4 result_color,
                            out float result_ao,
                            out float outdifference)
{
  result_ao = ambient_occlusion_eval(normal, dist, inverted, sample_count);
  result_color = result_ao * color;
  /* get depth data */
  vec3 vP = transform_point(ViewMatrix, g_data.P);
  float shadingdepth = abs(-vP.z);
  float zbuffer = abs(ambient_occlusion_depth());
  /* difference softness control */
  softness += 0.00001;
  shadingdepth = map(shadingdepth, 0.0, softness, 0.0, 1.0);
  zbuffer = map(zbuffer, 0.0, softness, 0.0, 1.0);
  /* out */
  outdifference = clamp(zbuffer - shadingdepth, 0.0, 1.0);
}
