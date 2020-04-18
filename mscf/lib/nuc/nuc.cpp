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
  if(abs(x) <= 1.0e-10) return 1.0/(2.0*n+1.0);
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

extern "C" void V_ijklmn(double *V, int I,int J,double *Ra, double *Rb, double ai, double bi,double *Rc_list, double *Zc_list, int nuc_num){
  double p = ai+bi;
  double Rp[3] = {(ai*Ra[0]+bi*Rb[0])/p, (ai*Ra[1]+bi*Rb[1])/p, (ai*Ra[2]+bi*Rb[2])/p};
  int IJ = I+J;
  double Rtuv[(IJ+1)*(IJ+1)*(IJ+1)] = {0.0};
  double rtuv[(IJ+1)*(IJ+1)*(IJ+1)] = {0.0};
  double Rc[3];
  double Zc;
  for(int nuc_i=0;nuc_i<nuc_num;nuc_i++){
    for(int i=0;i<3;i++){
      Rc[i] = Rc_list[nuc_i*3+i];
      Zc = Zc_list[nuc_i];
    }
    R_tuv(rtuv,I+J,Rp,Rc,p);
    for(int t=0;t<IJ+1;t++){
      for(int u=0;u<IJ+1-t;u++){
	for(int v=0;v<IJ+1-t-u;v++){
	  Rtuv[(t*(IJ+1)+u)*(IJ+1)+v] += Zc * rtuv[(t*(IJ+1)+u)*(IJ+1)+v];
	}
      }
    }
  }
  double Eijt[(I+1)*(J+1)*(IJ+1)]={0.0};
  double Eklu[(I+1)*(J+1)*(IJ+1)]={0.0};
  double Emnv[(I+1)*(J+1)*(IJ+1)]={0.0};
  E_ij_t(Eijt,I,J,Ra[0],Rb[0],ai,bi);
  E_ij_t(Eklu,I,J,Ra[1],Rb[1],ai,bi);
  E_ij_t(Emnv,I,J,Ra[2],Rb[2],ai,bi);
  int m,n;
  for(int i=0;i<I+1;i++){
    for(int j=0;j<J+1;j++){
      for(int k=0;k<I+1-i;k++){
	for(int l=0;l<J+1-j;l++){
	  m = I-i-k;
	  n = J-j-l;
	  V[((i*(J+1)+j)*(I+1)+k)*(J+1)+l] =0.0;
	  for(int t=0;t<i+j+1;t++){
	    for(int u=0;u<k+l+1;u++){
	      for(int v=0;v<m+n+1;v++){
		V[((i*(J+1)+j)*(I+1)+k)*(J+1)+l] += Eijt[(i*(J+1)+j)*(IJ+1)+t]*Eklu[(k*(J+1)+l)*(IJ+1)+u]*Emnv[(m*(J+1)+n)*(IJ+1)+v]*Rtuv[(t*(IJ+1)+u)*(IJ+1)+v];
	      }
	    }
	  }
	  V[((i*(J+1)+j)*(I+1)+k)*(J+1)+l] *= -2*M_PI/p;
	}
      }
    }
  }
}

extern "C" void cont_V1e(double *V, int I, int J, int P, int Q, double *Ra, double *Rb, double *a, double *b, double *da, double *db, double *Rc_list, double *Zc_list, int nuc_num) {
  double Vijkl[P*Q][(I+1)*(J+1)*(I+1)*(J+1)] = {0.0};
  double w_fact[] = {1.0,1.0,3.0,15.0,105.0};
  for(int p=0;p<P;p++){
    for(int q=0;q<Q;q++){
      V_ijklmn(Vijkl[p*Q+q], I, J, Ra, Rb, a[p], b[q], Rc_list, Zc_list,nuc_num);
    }
  }
  double ans;
  double Na,Nb;
  int m,n;
  for(int i=0;i<I+1;i++){
    for(int j=0;j<J+1;j++){
      for(int k=0;k<I+1-i;k++){
  	for(int l=0;l<J+1-j;l++){
  	  m = I-i-k;
  	  n = J-j-l;
  	  for(int p=0;p<P;p++){
  	    for(int q=0;q<Q;q++){
  	      ans = da[p]*db[q]*Vijkl[p*Q+q][((i*(J+1)+j)*(I+1)+k)*(J+1)+l];
  	      Na = pow(2.0*a[p]/M_PI,3.0/4.0)*sqrt(pow(4.0*a[p],I)/w_fact[I]);
  	      Nb = pow(2.0*b[q]/M_PI,3.0/4.0)*sqrt(pow(4.0*b[q],J)/w_fact[J]);
  	      ans *= Na * Nb;	      
  	      V[((i*(J+1)+j)*(I+1)+k)*(J+1)+l] += ans;
  	    }
  	  }
  	}
      }
    }
  }
}

extern "C" void V1e_lm(double *V, int la, int lb, int P, int Q, double *Ra, double *Rb, double *a, double *b, double *da, double *db, double *Rc_list, double *Zc_list, int nuc_num){
  double Vab[(la+1)*(lb+1)*(la+1)*(lb+1)] = {0.0};
  cont_V1e(Vab, la, lb, P, Q, Ra, Rb, a, b, da, db, Rc_list, Zc_list, nuc_num);
  int max_l = max(la, lb);
  int fact[max_l*2+1];
  fact[0] = 1;
  for(int i=1; i<max_l*2+1;i++) fact[i] = fact[i-1] * i;
  vector<vector<int>> comb(max_l+1,vector<int>(max_l+1,0));
  _comb(comb);
  double C_a[la+1][la/2+1][la/2+1][la+1];
  double C_b[lb+1][lb/2+1][lb/2+1][lb+1];
  for(int ma_=0;ma_<la+1;ma_++){
    for(int t=0;t<la/2+1;t++){
      for(int u=0;u<la/2+1;u++){
	for(int v=0;v<la+1;v++){
	  if(la>=t && t>=u && la-t>=ma_+t && ma_>=v){
	    C_a[ma_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[la][t]*comb[la-t][ma_+t]*comb[t][u]*comb[ma_][v]; 
	  }
	  else{
	    C_a[ma_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }
  for(int mb_=0;mb_<lb+1;mb_++){
    for(int t=0;t<lb/2+1;t++){
      for(int u=0;u<lb/2+1;u++){
	for(int v=0;v<lb+1;v++){
	  if(lb>=t && t>=u && lb-t>=mb_+t && mb_>=v){
	    C_b[mb_][t][u][v] = (-2.0*(t%2)+1)*(1.0/pow(4.0,t))*comb[lb][t]*comb[lb-t][mb_+t]*comb[t][u]*comb[mb_][v]; 
	  }
	  else{
	    C_b[mb_][t][u][v] = 0.0;
	  }
	}
      }
    }
  }
  int ma,mb,ma_,mb_,f;
  double Nma, Nmb;
  double va_, vb_;
  int pow_xa, pow_xb, pow_ya, pow_yb;
  for(int i=0;i<2*la+1;i++){
    for(int j=0;j<2*lb+1;j++){
      V[i*(2*lb+1)+j] = 0.0;
      ma = i-la; mb = j-lb;
      ma_ = abs(ma); mb_ = abs(mb);
      if(ma==0){
	Nma = (1.0/(pow(2,ma_)*fact[la]))*sqrt(1.0*fact[la+ma_]*fact[la-ma_]);
      }else{
	Nma = (1.0/(pow(2,ma_)*fact[la]))*sqrt(2.0*fact[la+ma_]*fact[la-ma_]);
      }
      if(mb==0){
	Nmb = (1.0/(pow(2,mb_)*fact[lb]))*sqrt(1.0*fact[lb+mb_]*fact[lb-mb_]);
      }else{
	Nmb = (1.0/(pow(2,mb_)*fact[lb]))*sqrt(2.0*fact[lb+mb_]*fact[lb-mb_]);
      }
      for(int ta=0;ta<(la-ma_)/2+1;ta++){
	for(int tb=0;tb<(lb-mb_)/2+1;tb++){
	  for(int ua=0;ua<ta+1;ua++){
	    for(int ub=0;ub<tb+1;ub++){
	      for(int va=0;va<ma_/2+1;va++){
		for(int vb=0;vb<mb_/2+1;vb++){
		  f = 1-2*((va+vb)%2);
		  va_ = va;
		  vb_ = vb;
		  if(ma<0){
		    va_ += 0.5;
		    if(va_ > (ma_-1)/2+0.5) break;
		  }
		  if(mb<0){
		    vb_ += 0.5;
		    if(vb_ > (mb_-1)/2+0.5) break;
		  }
		  pow_xa = floor(2*ta+ma_-2*(ua+va_));
		  pow_xb = floor(2*tb+mb_-2*(ub+vb_));
		  pow_ya = floor(2*(ua+va_));
		  pow_yb = floor(2*(ub+vb_));

		  V[i*(2*lb+1)+j] += f*C_a[ma_][ta][ua][int(2*va_)]*C_b[mb_][tb][ub][int(2*vb_)]*Vab[((pow_xa*(lb+1)+pow_xb)*(la+1)+pow_ya)*(lb+1)+pow_yb];
		}
	      }
	    }
	  }
	}
      }
      V[i*(2*lb+1)+j] *= Nma*Nmb;
    }
  }
}

extern "C" void get_v1e(double *V, double **R, int *l,double **a, double **da,int *P, int basis_len, int  basis_num, double *Rc_list, double *Zc_list, int nuc_num){
  int ind_i=0;
  int ind_j=0;
  int la, lb;
  int change[3][5] ={{0},{1,2,0},{0,1,2,3,4}};
  for(int i=0; i<basis_len;i++){
    ind_j = 0;
    for(int j=0;j<basis_len;j++){
      la = l[i]; lb = l[j];
      double Vlm[(la+1)*(lb+1)*(la+1)*(lb+1)] = {0.0};
      V1e_lm(Vlm, la, lb, P[i], P[j], R[i], R[j], a[i], a[j], da[i], da[j], Rc_list, Zc_list, nuc_num);
      for(int k=0;k<2*la+1;k++){
	for(int l=0;l<2*lb+1;l++){
	  V[(ind_i+change[la][k])*basis_num+ind_j+change[lb][l]] = Vlm[k*(2*lb+1)+l];
	}
      }
      ind_j+=2*lb+1;
    }
    ind_i+=2*la+1;
  }
}

