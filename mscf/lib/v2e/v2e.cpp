#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <boost/math/special_functions/gamma.hpp>

using namespace std;

extern "C" void _comb(vector<vector <int> > &v){
  for(int i = 0;i <v.size(); i++){
    v[i][0]=1;
    v[i][i]=1;
  }
  for(int k = 1;k <v.size();k++){
    for(int j = 1;j<k;j++){
      v[k][j]=(v[k-1][j-1]+v[k-1][j]);
    }
  }
}

extern "C" void E_ij_t(double *E, int I,int J, double Ax, double Bx, double ai, double bi){
  double p = ai+bi;
  double mu = ai*bi/p;
  double Xab = Ax-Bx;
  int IJ = I+J;
  E[(0*(J+1)+0)*(IJ+1)+0] = exp(-mu*pow(Xab,2));
  for(int j=0;j<J+1;j++){
    for(int i=0;i<I+1;i++){
      if (i==0 && j==0) continue;
      if(i==0){
	for(int t=0;t<i+j+1;t++){
	  E[(i*(J+1)+j)*(IJ+1)+t] = 0.0;
	  if(0<=t-1) E[(i*(J+1)+j)*(IJ+1)+t] += 1/(2*p)*E[(i*(J+1)+j-1)*(IJ+1)+t-1];
	  if(t<=i+j-1) E[(i*(J+1)+j)*(IJ+1)+t] += ai/p*Xab*E[(i*(J+1)+j-1)*(IJ+1)+t];
	  if(t+1<=i+j-1) E[(i*(J+1)+j)*(IJ+1)+t] +=(t+1)*E[(i*(J+1)+j-1)*(IJ+1)+t+1];
	}
      }else{
	for(int t=0;t<i+j+1;t++){
	  E[(i*(J+1)+j)*(IJ+1)+t] = 0.0;
	  if(0<=t-1) E[(i*(J+1)+j)*(IJ+1)+t] += 1/(2*p)*E[((i-1)*(J+1)+j)*(IJ+1)+t-1];
	  if(t<=i+j-1) E[(i*(J+1)+j)*(IJ+1)+t] += -bi/p*Xab*E[((i-1)*(J+1)+j)*(IJ+1)+t];
	  if(t+1<=i+j-1) E[(i*(J+1)+j)*(IJ+1)+t] += (t+1)*E[((i-1)*(J+1)+j)*(IJ+1)+t+1];
	}
      }
    }
  }
}		       

extern "C" double Fn(int n, double x){
  if(x <= 1.0e-1){
    double result = 1/(2*n+1.0);
    double result_k = 1.0;
    double result_x = 1.0;
    for(int k=1;k<7;k++){
      result_x *= (-x);
      result_k *= k;
      result += result_x/(result_k*(2*n+2*k+1));
    }
    return result;
  }
  double value_a = boost::math::tgamma(n+0.5);
  double value_b = boost::math::gamma_p(n+0.5, x);
  double value_c = 2*pow(x,n+0.5);
  return value_a*value_b/value_c;
}    

extern "C" void R_tuv(double *r,int IJ, double *Rp, double *Rc, double p){
  double R[IJ+1][(IJ+1)*(IJ+1)*(IJ+1)];
  double Xpc = Rp[0] - Rc[0];
  double Ypc = Rp[1] - Rc[1];
  double Zpc = Rp[2] - Rc[2];
  double Rpc_2 = pow(Xpc,2) + pow(Ypc,2) + pow(Zpc,2);
  for(int n=0;n<IJ+1;n++){
    R[n][0] = pow(-2*p,n) * Fn(n, p*Rpc_2);
  }
  for(int n=IJ-1;n>=0;n--){
    for(int t=0;t<IJ+1-n;t++){
      for(int u=0;u<IJ+1-n-t;u++){
	for(int v=0;v<IJ+1-n-t-u;v++){
	  if(t+u+v!=0) R[n][(t*(IJ+1)+u)*(IJ+1)+v] = 0.0;
	  if(t>=1){
	    if(t>=2) R[n][(t*(IJ+1)+u)*(IJ+1)+v] += (t-1)*R[n+1][((t-2)*(IJ+1)+u)*(IJ+1)+v];
	    R[n][(t*(IJ+1)+u)*(IJ+1)+v] += Xpc * R[n+1][((t-1)*(IJ+1)+u)*(IJ+1)+v];
	  }else if(u>=1){
	    if(u>=2) R[n][(t*(IJ+1)+u)*(IJ+1)+v] += (u-1)*R[n+1][(t*(IJ+1)+u-2)*(IJ+1)+v];
	    R[n][(t*(IJ+1)+u)*(IJ+1)+v] += Ypc * R[n+1][(t*(IJ+1)+u-1)*(IJ+1)+v];
	  }else if(v>=1){
	    if(v>=2) R[n][(t*(IJ+1)+u)*(IJ+1)+v] += (v-1)*R[n+1][(t*(IJ+1)+u)*(IJ+1)+v-2];
	    R[n][(t*(IJ+1)+u)*(IJ+1)+v] += Zpc * R[n+1][(t*(IJ+1)+u)*(IJ+1)+v-1];
	  }
	}
      }
    }
  }
  for(int t=0;t<IJ+1;t++){
    for(int u=0;u<IJ+1-t;u++){
      for(int v=0;v<IJ+1-t-u;v++){
	r[(t*(IJ+1)+u)*(IJ+1)+v] = R[0][(t*(IJ+1)+u)*(IJ+1)+v];
      }
    }
  }
}

extern "C" void g_abcd(double *V, int I1, int J1, int I2, int J2, double *Ra, double *Rb, double *Rc, double *Rd, double ai, double bi, double ci, double di){
  double p = ai+bi;
  double q = ci+di;
  int IJ1 = I1+J1;
  int IJ2 = I2+J2;
  int IJIJ = IJ1+IJ2;
  double alpha = p*q/(p+q);
  double Rp[3] = {(ai*Ra[0]+bi*Rb[0])/p, (ai*Ra[1]+bi*Rb[1])/p, (ai*Ra[2]+bi*Rb[2])/p};
  double Rq[3] = {(ci*Rc[0]+di*Rd[0])/q, (ci*Rc[1]+di*Rd[1])/q, (ci*Rc[2]+di*Rd[2])/q};
  
  double Rtuv[(I1+J1+I2+J2+1)*(I1+J1+I2+J2+1)*(I1+J1+I2+J2+1)] = {0.0};
  R_tuv(Rtuv, I1+J1+I2+J2, Rp, Rq, alpha);

  double Eijt1[(I1+1)*(J1+1)*(I1+J1+1)] = {0.0};
  double Eklu1[(I1+1)*(J1+1)*(I1+J1+1)] = {0.0};
  double Emnv1[(I1+1)*(J1+1)*(I1+J1+1)] = {0.0};
  E_ij_t(Eijt1, I1, J1, Ra[0], Rb[0], ai, bi);
  E_ij_t(Eklu1, I1, J1, Ra[1], Rb[1], ai, bi);
  E_ij_t(Emnv1, I1, J1, Ra[2], Rb[2], ai, bi);

  double Eijt2[(I2+1)*(J2+1)*(I2+J2+1)] = {0.0};
  double Eklu2[(I2+1)*(J2+1)*(I2+J2+1)] = {0.0};
  double Emnv2[(I2+1)*(J2+1)*(I2+J2+1)] = {0.0};
  E_ij_t(Eijt2, I2, J2, Rc[0], Rd[0], ci, di);
  E_ij_t(Eklu2, I2, J2, Rc[1], Rd[1], ci, di);
  E_ij_t(Emnv2, I2, J2, Rc[2], Rd[2], ci, di);
  int m1, n1, m2, n2,f;
  double Etuv, ans1,ans2;
  for(int i1=0;i1<I1+1;i1++){
    for(int j1=0;j1<J1+1;j1++){
      for(int k1=0;k1<I1+1-i1;k1++){
	for(int l1=0;l1<J1+1-j1;l1++){
	  m1 = I1-i1-k1;
	  n1 = J1-j1-l1;
	  for(int i2=0;i2<I2+1;i2++){
	    for(int j2=0;j2<J2+1;j2++){
	      for(int k2=0;k2<I2+1-i2;k2++){
		for(int l2=0;l2<J2+1-j2;l2++){
		  m2 = I2-i2-k2;
		  n2 = J2-j2-l2;
		  ans1 = 0.0;
		  for(int t1=0;t1<i1+j1+1;t1++){
		    for(int u1=0;u1<k1+l1+1;u1++){
		      for(int v1=0;v1<m1+n1+1;v1++){
			Etuv = Eijt1[(i1*(J1+1)+j1)*(IJ1+1)+t1]*Eklu1[(k1*(J1+1)+l1)*(IJ1+1)+u1]*Emnv1[(m1*(J1+1)+n1)*(IJ1+1)+v1];
			ans2 = 0.0;
			for(int t2=0;t2<i2+j2+1;t2++){
			  for(int u2=0;u2<k2+l2+1;u2++){
			    for(int v2=0;v2<m2+n2+1;v2++){
			      f = 1-2*((t2+u2+v2)%2);
			      ans2 += f*Eijt2[(i2*(J2+1)+j2)*(IJ2+1)+t2]*Eklu2[(k2*(J2+1)+l2)*(IJ2+1)+u2]*Emnv2[(m2*(J2+1)+n2)*(IJ2+1)+v2]*Rtuv[((t1+t2)*(IJIJ+1)+u1+u2)*(IJIJ+1)+v1+v2];
			    }
			  }
			}
			ans1 += ans2 * Etuv;
		      }
		    }
		  }
		  V[((((((i1*(J1+1)+j1)*(I1+1)+k1)*(J1+1)+l1)*(I2+1)+i2)*(J2+1)+j2)*(I2+1)+k2)*(J2+1)+l2] = ans1*(2*pow(M_PI,2.5)/(p*q*sqrt(p+q)));
		}
	      }
	    }
	  }
	}
      }
    }
  }
}


extern "C" void cont_V2e(double *V,int I1,int J1,int I2,int J2,int P1,int Q1,int P2,int Q2,double *Ra,double *Rb,double *Rc,double *Rd,double *a,double *b,double *c,double *d,double *da,double *db,double *dc,double *dd){
  double gabcd[P1*Q1*P2*Q2][(I1+1)*(J1+1)*(I1+1)*(J1+1)*(I2+1)*(J2+1)*(I2+1)*(J2+1)];
  double w_fact[] = {1.0,1.0,3.0,15.0,105.0};
  for(int p1=0;p1<P1;p1++){
    for(int q1=0;q1<Q1;q1++){
      for(int p2=0;p2<P2;p2++){
	for(int q2=0;q2<Q2;q2++){
	  g_abcd(gabcd[((p1*Q1+q1)*P2+p2)*Q2+q2],I1,J1,I2,J2,Ra,Rb,Rc,Rd,a[p1],b[q1],c[p2],d[q2]);
	}
      }
    }
  }
  double ans1, ans2;
  double Na,Nb,Nc,Nd;
  for(int i1=0;i1<I1+1;i1++){
    for(int j1=0;j1<J1+1;j1++){
      for(int k1=0;k1<I1+1-i1;k1++){
	for(int l1=0;l1<J1+1-j1;l1++){
	  for(int i2=0;i2<I2+1;i2++){
	    for(int j2=0;j2<J2+1;j2++){
	      for(int k2=0;k2<I2+1-i2;k2++){
		for(int l2=0;l2<J2+1-j2;l2++){
		  ans1 = 0.0;
		  for(int p1=0;p1<P1;p1++){
		    for(int q1=0;q1<Q1;q1++){
		      for(int p2=0;p2<P2;p2++){
			for(int q2=0;q2<Q2;q2++){
			  ans2 = da[p1]*db[q1]*dc[p2]*dd[q2];
			  ans2 *= gabcd[((p1*Q1+q1)*P2+p2)*Q2+q2][((((((i1*(J1+1)+j1)*(I1+1)+k1)*(J1+1)+l1)*(I2+1)+i2)*(J2+1)+j2)*(I2+1)+k2)*(J2+1)+l2];
			  Na = pow(2.0*a[p1]/M_PI,3.0/4.0)*sqrt(pow(4.0*a[p1],I1)/w_fact[I1]);
			  Nb = pow(2.0*b[q1]/M_PI,3.0/4.0)*sqrt(pow(4.0*b[q1],J1)/w_fact[J1]);
			  Nc = pow(2.0*c[p2]/M_PI,3.0/4.0)*sqrt(pow(4.0*c[p2],I2)/w_fact[I2]);
			  Nd = pow(2.0*d[q2]/M_PI,3.0/4.0)*sqrt(pow(4.0*d[q2],J2)/w_fact[J2]);
			  ans1 += Na*Nb*Nc*Nd*ans2;
			}
		      }
		    }
		  }
		  V[((((((i1*(J1+1)+j1)*(I1+1)+k1)*(J1+1)+l1)*(I2+1)+i2)*(J2+1)+j2)*(I2+1)+k2)*(J2+1)+l2] = ans1;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}

extern "C" void V2e_lm(double *V,int I1,int J1,int I2,int J2,int P1,int Q1,int P2,int Q2,double *Ra,double *Rb,double *Rc,double *Rd,double *a,double *b,double *c,double *d,double *da,double *db,double *dc,double *dd){
  double Vabcd[(I1+1)*(J1+1)*(I1+1)*(J1+1)*(I2+1)*(J2+1)*(I2+1)*(J2+1)];
  cont_V2e(Vabcd,I1,J1,I2,J2,P1,Q1,P2,Q2,Ra,Rb,Rc,Rd,a,b,c,d,da,db,dc,dd);
  int max_l = max({I1, J1, I2, J2});
  int fact[max_l*2+1];
  fact[0] = 1;
  for(int i=1; i<max_l*2+1;i++) fact[i] = fact[i-1] * i;
  vector<vector<int>> comb(max_l+1,vector<int>(max_l+1,0));
  _comb(comb);
  double C_a[I1+1][I1/2+1][I1/2+1][I1+1];
  double C_b[J1+1][J1/2+1][J1/2+1][J1+1];
  double C_c[I2+1][I2/2+1][I2/2+1][I2+1];
  double C_d[J2+1][J2/2+1][J2/2+1][J2+1];
  for(int ma_=0;ma_<I1+1;ma_++){
    for(int t=0;t<I1/2+1;t++){
      for(int u=0;u<I1/2+1;u++){
	for(int v=0;v<I1+1;v++){
	  if(I1>=t && t>=u && I1-t>=ma_+t && ma_>=v){
	    C_a[ma_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[I1][t]*comb[I1-t][ma_+t]*comb[t][u]*comb[ma_][v]; 
	  }
	  else{
	    C_a[ma_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }
  
  for(int mb_=0;mb_<J1+1;mb_++){
    for(int t=0;t<J1/2+1;t++){
      for(int u=0;u<J1/2+1;u++){
	for(int v=0;v<J1+1;v++){
	  if(J1>=t && t>=u && J1-t>=mb_+t && mb_>=v){
	    C_b[mb_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[J1][t]*comb[J1-t][mb_+t]*comb[t][u]*comb[mb_][v]; 
	  }
	  else{
	    C_b[mb_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }

  for(int mc_=0;mc_<I2+1;mc_++){
    for(int t=0;t<I2/2+1;t++){
      for(int u=0;u<I2/2+1;u++){
	for(int v=0;v<I2+1;v++){
	  if(I2>=t && t>=u && I2-t>=mc_+t && mc_>=v){
	    C_c[mc_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[I2][t]*comb[I2-t][mc_+t]*comb[t][u]*comb[mc_][v]; 
	  }
	  else{
	    C_c[mc_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }
  
  for(int md_=0;md_<J2+1;md_++){
    for(int t=0;t<J2/2+1;t++){
      for(int u=0;u<J2/2+1;u++){
	for(int v=0;v<J2+1;v++){
	  if(J2>=t && t>=u && J2-t>=md_+t && md_>=v){
	    C_d[md_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[J2][t]*comb[J2-t][md_+t]*comb[t][u]*comb[md_][v]; 
	  }
	  else{
	    C_d[md_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }
  
  int ma,mb,mc,md,ma_,mb_,mc_,md_,f;
  double Nma, Nmb, Nmc, Nmd;
  double va_, vb_,vc_,vd_;
  int pow_xa,pow_xb,pow_xc,pow_xd,pow_ya,pow_yb,pow_yc,pow_yd;
  double ans;
  for(int i=0;i<2*I1+1;i++){
    for(int j=0;j<2*J1+1;j++){
      for(int k=0;k<2*I2+1;k++){
	for(int l=0;l<2*J2+1;l++){
	  ans = 0.0;
	  ma = i-I1; mb = j-J1; mc = k-I2; md = l-J2;
	  ma_ = abs(ma); mb_ = abs(mb); mc_ = abs(mc); md_ = abs(md);
	  if(ma==0){
	    Nma = (1.0/(pow(2,ma_)*fact[I1]))*sqrt(1.0*fact[I1+ma_]*fact[I1-ma_]);
	  }else{
	    Nma = (1.0/(pow(2,ma_)*fact[I1]))*sqrt(2.0*fact[I1+ma_]*fact[I1-ma_]);
	  }
	  if(mb==0){
	    Nmb = (1.0/(pow(2,mb_)*fact[J1]))*sqrt(1.0*fact[J1+mb_]*fact[J1-mb_]);
	  }else{
	    Nmb = (1.0/(pow(2,mb_)*fact[J1]))*sqrt(2.0*fact[J1+mb_]*fact[J1-mb_]);
	  }
	  if(mc==0){
	    Nmc = (1.0/(pow(2,mc_)*fact[I2]))*sqrt(1.0*fact[I2+mc_]*fact[I2-mc_]);
	  }else{
	    Nmc = (1.0/(pow(2,mc_)*fact[I2]))*sqrt(2.0*fact[I2+mc_]*fact[I2-mc_]);
	  }
	  if(md==0){
	    Nmd = (1.0/(pow(2,md_)*fact[J2]))*sqrt(1.0*fact[J2+md_]*fact[J2-md_]);
	  }else{
	    Nmd = (1.0/(pow(2,md_)*fact[J2]))*sqrt(2.0*fact[J2+md_]*fact[J2-md_]);
	  }
	  for(int ta=0;ta<(I1-ma_)/2+1;ta++){
	    for(int tb=0;tb<(J1-mb_)/2+1;tb++){
	      for(int tc=0;tc<(I2-mc_)/2+1;tc++){
		for(int td=0;td<(J2-md_)/2+1;td++){
		  for(int ua=0;ua<ta+1;ua++){
		    for(int ub=0;ub<tb+1;ub++){
		      for(int uc=0;uc<tc+1;uc++){
			for(int ud=0;ud<td+1;ud++){
			  for(int va=0;va<ma_/2+1;va++){
			    for(int vb=0;vb<mb_/2+1;vb++){
			      for(int vc=0;vc<mc_/2+1;vc++){
				for(int vd=0;vd<md_/2+1;vd++){
				  f = 1-2*((va+vb+vc+vd)%2);
				  va_ = va; vb_ = vb; vc_ = vc; vd_ = vd;
				  if(ma<0){
				    va_ += 0.5;
				    if(va_ > (ma_-1)/2+0.5) break;
				  }
				  if(mb<0){
				    vb_ += 0.5;
				    if(vb_ > (mb_-1)/2+0.5) break;
				  }
				  if(mc<0){
				    vc_ += 0.5;
				    if(vc_ > (mc_-1)/2+0.5) break;
				  }
				  if(md<0){
				    vd_ += 0.5;
				    if(vd_ > (md_-1)/2+0.5) break;
				  }
				  pow_xa = floor(2*ta+ma_-2*(ua+va_));
				  pow_xb = floor(2*tb+mb_-2*(ub+vb_));
				  pow_xc = floor(2*tc+mc_-2*(uc+vc_));
				  pow_xd = floor(2*td+md_-2*(ud+vd_));
				  pow_ya = floor(2*(ua+va_));
				  pow_yb = floor(2*(ub+vb_));
				  pow_yc = floor(2*(uc+vc_));
				  pow_yd = floor(2*(ud+vd_));
				  ans += f*C_a[ma_][ta][ua][int(2*va_)]*C_b[mb_][tb][ub][int(2*vb_)]*C_c[mc_][tc][uc][int(2*vc_)]*C_d[md_][td][ud][int(2*vd_)]*Vabcd[((((((pow_xa*(J1+1)+pow_xb)*(I1+1)+pow_ya)*(J1+1)+pow_yb)*(I2+1)+pow_xc)*(J2+1)+pow_xd)*(I2+1)+pow_yc)*(J2+1)+pow_yd];
				}
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	  V[((i*(2*J1+1)+j)*(2*I2+1)+k)*(2*J2+1)+l] = ans*Nma*Nmb*Nmc*Nmd;
	}
      }
    }
  }
}

extern "C" void get_v2e(double *V, double **R, int *L,double **a, double **da,int *P, int basis_len, int  basis_num){
  int ind_i, ind_j, ind_k, ind_l;
  int ind_I, ind_J, ind_K, ind_L;
  int I1, J1, I2, J2;
  int change[3][5] ={{0},{1,2,0},{0,1,2,3,4}};
  bool check[basis_len][basis_len][basis_len][basis_len];
  
  for(int i=0;i<basis_len;i++){
    for(int j=0;j<basis_len;j++){
      for(int k=0;k<basis_len;k++){
	for(int l=0;l<basis_len;l++){
	  check[i][j][k][l] = false;
	}
      }
    }
  }
  
  ind_i = 0;
  double ans;
  for(int i=0;i<basis_len;i++){
    ind_j = 0;
    for(int j=0;j<basis_len;j++){
      ind_k = 0;
      for(int k=0;k<basis_len;k++){
	ind_l = 0;
	for(int l=0;l<basis_len;l++){
	  I1 = L[i]; J1 = L[j]; I2 = L[k]; J2 = L[l];
	  if(!(check[i][j][k][l])){
	    check[i][j][k][l] = true; check[i][j][l][k] = true; check[j][i][k][l] = true; check[j][i][l][k] = true;
	    check[k][l][i][j] = true; check[k][l][j][i] = true; check[l][k][i][j] = true; check[l][k][j][i] = true;
	    double V2elm[(2*I1+1)*(2*J1+1)*(2*I2+1)*(2*J2+1)] = {0.0};
	    V2e_lm(V2elm,I1,J1,I2,J2,P[i],P[j],P[k],P[l],R[i],R[j],R[k],R[l],a[i],a[j],a[k],a[l],da[i],da[j],da[k],da[l]);
	    for(int ma=0;ma<2*I1+1;ma++){
	      for(int mb=0;mb<2*J1+1;mb++){
		for(int mc=0;mc<2*I2+1;mc++){
		  for(int md=0;md<2*J2+1;md++){
		    ans = V2elm[((ma*(2*J1+1)+mb)*(2*I2+1)+mc)*(2*J2+1)+md];
		    ind_I = ind_i + change[I1][ma];
		    ind_J = ind_j + change[J1][mb];
		    ind_K = ind_k + change[I2][mc];
		    ind_L = ind_l + change[J2][md];

		    V[((ind_I*basis_num+ind_J)*basis_num+ind_K)*basis_num+ind_L] = ans; V[((ind_I*basis_num+ind_J)*basis_num+ind_L)*basis_num+ind_K] = ans;
		    V[((ind_J*basis_num+ind_I)*basis_num+ind_K)*basis_num+ind_L] = ans; V[((ind_J*basis_num+ind_I)*basis_num+ind_L)*basis_num+ind_K] = ans;

		    V[((ind_K*basis_num+ind_L)*basis_num+ind_I)*basis_num+ind_J] = ans; V[((ind_K*basis_num+ind_L)*basis_num+ind_J)*basis_num+ind_I] = ans;
		    V[((ind_L*basis_num+ind_K)*basis_num+ind_I)*basis_num+ind_J] = ans; V[((ind_L*basis_num+ind_K)*basis_num+ind_J)*basis_num+ind_I] = ans;
		  }
		}
	      }
	    }
	  }
	  ind_l += 2*J2+1;
	}
	ind_k += 2*I2+1;
      }
      ind_j += 2*J1+1;
    }
    ind_i += 2*I1+1;
  }
}
	  
         
