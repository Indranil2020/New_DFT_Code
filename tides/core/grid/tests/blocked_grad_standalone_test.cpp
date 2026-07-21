// Validate AddBlockedGrad + BlockedRhoWithGrad (Phase 3 Inc 3) vs dense.
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
  std::vector<double> rcut(centers.size(),r_cut);
  std::vector<double> alpha={0.5,0.7,0.4,0.6};
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
    double f=std::exp(-alpha[i]*r2), k=-2.0*alpha[i]*f; return {k*dx,k*dy,k*dz};};
  auto bp=BuildBlockedPhi(grid,centers,rcut,eval,8);
  AddBlockedGrad(grid,bp,grad_eval);
  // dense phi + grad
  std::vector<std::vector<double>> phi(n,std::vector<double>(n_grid,0.0));
  std::array<std::vector<std::vector<double>>,3> gph;
  for(int c=0;c<3;++c) gph[c].assign(n,std::vector<double>(n_grid,0.0));
  for(std::size_t i=0;i<n;++i)for(std::size_t iz=0;iz<grid.n[2];++iz)for(std::size_t iy=0;iy<grid.n[1];++iy)for(std::size_t ix=0;ix<grid.n[0];++ix){
    auto c=grid.coord(ix,iy,iz); std::size_t g=grid.flatten(ix,iy,iz);
    phi[i][g]=eval(i,c[0],c[1],c[2]); auto gg=grad_eval(i,c[0],c[1],c[2]);
    for(int d=0;d<3;++d) gph[d][i][g]=gg[d];}
  std::mt19937 rng(11); std::uniform_real_distribution<double> U(-1,1);
  std::vector<double> P(n*n); for(std::size_t i=0;i<n;++i)for(std::size_t j=i;j<n;++j)P[i*n+j]=P[j*n+i]=U(rng);
  // dense rho + grad (factor-2 convention)
  std::vector<double> rho_ref(n_grid,0.0); std::array<std::vector<double>,3> gr_ref;
  for(int c=0;c<3;++c) gr_ref[c].assign(n_grid,0.0);
  std::vector<double> temp(n*n_grid,0.0);
  for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j){double Pij=P[i*n+j];
    for(std::int64_t g=0;g<n_grid;++g) temp[i*n_grid+g]+=Pij*phi[j][g];}
  for(std::size_t i=0;i<n;++i)for(std::int64_t g=0;g<n_grid;++g){
    rho_ref[g]+=phi[i][g]*temp[i*n_grid+g];
    for(int c=0;c<3;++c) gr_ref[c][g]+=2.0*gph[c][i][g]*temp[i*n_grid+g];}
  auto rg=BlockedRhoWithGrad(grid,bp,P);
  double dr=0,dgx=0,dgy=0,dgz=0;
  for(std::int64_t g=0;g<n_grid;++g){dr=std::max(dr,std::fabs(rho_ref[g]-rg.rho[g]));
    dgx=std::max(dgx,std::fabs(gr_ref[0][g]-rg.grad_x[g]));
    dgy=std::max(dgy,std::fabs(gr_ref[1][g]-rg.grad_y[g]));
    dgz=std::max(dgz,std::fabs(gr_ref[2][g]-rg.grad_z[g]));}
  std::printf("|rho|=%.2e |grad_x|=%.2e |grad_y|=%.2e |grad_z|=%.2e  %s\n",dr,dgx,dgy,dgz,
    (dr<1e-12&&dgx<1e-12&&dgy<1e-12&&dgz<1e-12)?"OK":"*** FAIL ***");
  return 0;
}
