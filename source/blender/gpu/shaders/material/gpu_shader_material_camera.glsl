void camera(out vec3 outview, out float outdepth, out float outdist)
//void camera(out vec3 outview, out float outdepth, out float outdist, out float outdifference)
{
  vec3 vP = transform_point(ViewMatrix, g_data.P);
  vP.z = -vP.z;
  outdepth = abs(vP.z);
  outdist = length(vP);
  outview = normalize(vP);

  /*---- todo redo zbuffer output depth - clean method broken on codegen refractor ----- */
  /* add data via GPU_link or do it on it's own node */ 

  //outdifference = ambient_occlusion_depth();
  
  /* todo calculate difference between shading point and zbuffer and output the difference only */
}
