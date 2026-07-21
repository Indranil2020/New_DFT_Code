// Validate BlockedGgaVmat (Phase 3 Inc 3b) vs dense BuildGgaHmatGemm formula.
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include "grid/grid_blocking.hpp"
#include "grid/blocked_ops.hpp"
using namespace tides::grid;
int main(){
  const double h=0.35, r_cut=5.0, margin=5.0;
  std::vector<std::array<double,3>> centers={{0,0,0},{2.6,0,0},{1.3,1.8,0},{1.3,-1.0,1.4}};
  std::vector<double> rcut(centers.size(),r_cut), alpha={0.5,0.7,0.4,0.6};
  const std::size_t n=centers.size();
  double xmin=-margin, xmax=2.6+margin;
  UniformGrid3D grid; grid.n={(std::size_t)((xmax-xmin)/h)+1,(std::size_t)((2*margin)/h)+1,(std::size_t)((2*margin)/h)+1};
  grid.h={h,h,h}; grid.origin={xmin,-margin,-margin};
  const std::int64_t n_grid=(std::int64_t)grid.n[0]*grid.n[1]*grid.n[2];
  auto eval=[&](std::size_t i,double x,double y,double z)->double{
    double dx=x-centers[i][0],dy=y-centers[i][1],dz=z-centers[i][2],r2=dx*dx+dy*dy+dz*dz;
    if(std::sqrt(r2)>rcut[i]) return 0.0; return std::exp(-alpha[i]*r2);};
  auto grad_eval=[&](std::size_t i,double x,double y,double z)->std::array<double,3>{
    double dx=x-centers[i][0],dy=y-centers[i][1],dz=z-centers[i][2],r2=dx*dx+dy*dy+dz*dz;
    if(std::sqrt(r2)>rcut[i]) return {0,0,0};
    double f=std::exp(-alpha[i]*r2),k=-2.0*alpha[i]*f; return {k*dx,k*dy,k*dz};};
  auto bp=BuildBlockedPhi(grid,centers,rcut,eval,8); AddBlockedGrad(grid,bp,grad_eval);
  // dense phi/grad
  std::vector<std::vector<double>> phi(n,std::vector<double>(n_grid,0.0));
  std::array<std::vector<std::vector<double>>,3> gph; for(int c=0;c<3;++c) gph[c].assign(n,std::vector<double>(n_grid,0.0));
  for(std::size_t i=0;i<n;++i)for(std::size_t iz=0;iz<grid.n[2];++iz)for(std::size_t iy=0;iy<grid.n[1];++iy)for(std::size_t ix=0;ix<grid.n[0];++ix){
    auto c=grid.coord(ix,iy,iz); std::size_t g=grid.flatten(ix,iy,iz);
    phi[i][g]=eval(i,c[0],c[1],c[2]); auto gg=grad_eval(i,c[0],c[1],c[2]); for(int d=0;d<3;++d) gph[d][i][g]=gg[d];}
  std::mt19937 rng(3); std::uniform_real_distribution<double> U(-1,1);
  std::vector<double> wr(n_grid),wx(n_grid),wy(n_grid),wz(n_grid);
  for(std::int64_t g=0;g<n_grid;++g){wr[g]=U(rng);wx[g]=U(rng);wy[g]=U(rng);wz[g]=U(rng);}
  // dense reference
  std::vector<double> Href(n*n,0.0);
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){double s=0.0;
    for(std::int64_t g=0;g<n_grid;++g) s+= wr[g]*phi[i][g]*phi[j][g]
      + wx[g]*(gph[0][i][g]*phi[j][g]+phi[i][g]*gph[0][j][g])
      + wy[g]*(gph[1][i][g]*phi[j][g]+phi[i][g]*gph[1][j][g])
      + wz[g]*(gph[2][i][g]*phi[j][g]+phi[i][g]*gph[2][j][g]);
    Href[i*n+j]=s;}
  auto Hb=BlockedGgaVmat(grid,bp,wr,wx,wy,wz);
  double dH=0; for(std::size_t k=0;k<n*n;++k) dH=std::max(dH,std::fabs(Href[k]-Hb[k]));
  std::printf("|GGA_vmat_blk - dense| = %.2e  %s\n",dH,dH<1e-12?"OK":"*** FAIL ***");
  return 0;
}
