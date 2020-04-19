#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

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

extern "C" void S_ij(double *S, int I, int J, double Ax, double Bx, double ai, double bi) {
  double p = ai + bi;
  double mu = ai * bi / p;
  double Xab = Ax - Bx;
  S[0*(J+1)+0] = sqrt(M_PI / p) * exp(-mu * pow(Xab,2));
  for(int i=0; i < I; i++) {
    S[(i+1)*(J+1)+0] = (-bi / p) * Xab * S[i*(J+1)+0];
    if (i != 0) {
      S[(i+1)*(J+1)+0] += (1/(2.0*p)) * (i * S[(i-1)*(J+1)+0]);
    }
  }
  for(int j=0; j<J; j++){
    for(int i=0; i<I+1; i++){
      S[i*(J+1)+j+1] = (ai / p) * Xab * S[i*(J+1)+j];
      if (i!= 0){
	S[i*(J+1)+j+1] += (1/(2.0*p)) * i*S[(i-1)*(J+1)+j];
      }
      if (j!=0) {
	S[i*(J+1)+j+1] += (1/(2.0*p)) * j*S[i*(J+1)+j-1];
      }
    }
  }
}

extern "C" void T_ij(double *Tij, double *Sij, int I, int J, double Ax, double Bx, double ai, double bi){
  double p = ai + bi;
  double Xab = Ax - Bx;
  S_ij(Sij, I, J, Ax, Bx, ai, bi);
  Tij[0*(J+1)+0] = (ai-2*pow(ai,2)*(pow((-bi*Xab/p),2)+1/(2*p)))*Sij[0*(J+1)+0];
  for(int i=0;i<I;i++){
    Tij[(i+1)*(J+1)+0] = -bi/p*Xab*Tij[i*(J+1)+0] + bi/p*2*ai*Sij[(i+1)*(J+1)+0];
    if(i!=0){
      Tij[(i+1)*(J+1)+0] += 1/(2*p)*i*Tij[(i-1)*(J+1)+0] - bi/p*i*Sij[(i-1)*(J+1)+0];
    }
  }
  for(int j=0;j<J;j++){
    for(int i=0;i<I+1;i++){
      Tij[i*(J+1)+j+1] = ai/p*Xab*Tij[i*(J+1)+j] + ai/p*2*bi*Sij[i*(J+1)+j+1];
      if(i!=0){
	Tij[i*(J+1)+j+1] += 1/(2*p)*i*Tij[(i-1)*(J+1)+j];
      }
      if(j!=0){
	Tij[i*(J+1)+j+1] += 1/(2*p)*j*Tij[i*(J+1)+j-1] - ai/p*j*Sij[i*(J+1)+j-1];
      }
    }
  }
}

extern "C" void cont_Tij(double *T, int I, int J, int P, int Q, double *Ra, double *Rb, double *a, double *b, double *da, double *db) {
  double Sij[P*Q][(I+1)*(J+1)];
  double Skl[P*Q][(I+1)*(J+1)];
  double Smn[P*Q][(I+1)*(J+1)];

  double Tij[P*Q][(I+1)*(J+1)];
  double Tkl[P*Q][(I+1)*(J+1)];
  double Tmn[P*Q][(I+1)*(J+1)];

  double w_fact[] = {1.0,1.0,3.0,15.0,105.0};
  for(int i=0;i<P;i++){
    for(int j=0;j<Q;j++){
      T_ij(Tij[i*Q+j],Sij[i*Q+j], I, J, Ra[0], Rb[0], a[i], b[j]);
      T_ij(Tkl[i*Q+j],Skl[i*Q+j], I, J, Ra[1], Rb[1], a[i], b[j]);
      T_ij(Tmn[i*Q+j],Smn[i*Q+j], I, J, Ra[2], Rb[2], a[i], b[j]);
    }
  }
  double ans;
  double Na,Nb;
  for(int i=0;i<I+1;i++){
    for(int j=0;j<J+1;j++){
      for(int k=0;k<I+1-i;k++){
	for(int l=0;l<J+1-j;l++){
	  int m = I-i-k;
	  int n = J-j-l;
	  for(int p=0;p<P;p++){
	    for(int q=0;q<Q;q++){
	      ans = Tij[p*Q+q][i*(J+1)+j]*Skl[p*Q+q][k*(J+1)+l]*Smn[p*Q+q][m*(J+1)+n];
	      ans += Sij[p*Q+q][i*(J+1)+j]*Tkl[p*Q+q][k*(J+1)+l]*Smn[p*Q+q][m*(J+1)+n];
	      ans += Sij[p*Q+q][i*(J+1)+j]*Skl[p*Q+q][k*(J+1)+l]*Tmn[p*Q+q][m*(J+1)+n];
	      
	      Na = pow(2.0*a[p]/M_PI,3.0/4.0)*sqrt(pow(4.0*a[p],I)/w_fact[I]);
	      Nb = pow(2.0*b[q]/M_PI,3.0/4.0)*sqrt(pow(4.0*b[q],J)/w_fact[J]);
	      ans *= da[p]*db[q]*Na * Nb;
	      
	      T[((i*(J+1)+j)*(I+1)+k)*(J+1)+l] += ans;
	    }
	  }
	}
      }
    }
  }  
}

extern "C" void T_lm(double *T, int la, int lb, int P, int Q, double *Ra, double *Rb, double *a, double *b, double *da, double *db){
  double Tab[(la+1)*(lb+1)*(la+1)*(lb+1)] = {0.0};
  cont_Tij(Tab, la, lb, P, Q, Ra, Rb, a, b, da, db);
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
      T[i*(2*lb+1)+j] = 0.0;
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
		  T[i*(2*lb+1)+j] += f*C_a[ma_][ta][ua][int(2*va_)]*C_b[mb_][tb][ub][int(2*vb_)]*Tab[((pow_xa*(lb+1)+pow_xb)*(la+1)+pow_ya)*(lb+1)+pow_yb];
		}
	      }
	    }
	  }
	}
      }
      T[i*(2*lb+1)+j] *= Nma*Nmb;
    }
  }
}

extern "C" void get_kin(double *T, double **R, int *l,double **a, double **da,int *P, int basis_len, int  basis_num){
  int ind_i=0;
  int ind_j=0;
  int la, lb;
  int change[3][5] ={{0},{1,2,0},{0,1,2,3,4}};
  bool check[basis_len][basis_len];
  
  for(int i=0;i<basis_len;i++){
    for(int j=0;j<basis_len;j++){
      check[i][j] = false;
    }
  }
  
  for(int i=0; i<basis_len;i++){
    ind_j = 0;
    for(int j=0;j<basis_len;j++){
      la = l[i]; lb = l[j];
      if(!(check[i][j])){
	check[i][j] = true;
	check[j][i] = true;
	double Tlm[(2*la+1)*(2*lb+1)] = {0.0};
	T_lm(Tlm, la, lb, P[i], P[j], R[i], R[j], a[i], a[j], da[i], da[j]);
	for(int k=0;k<2*la+1;k++){
	  for(int l=0;l<2*lb+1;l++){
	    T[(ind_i+change[la][k])*basis_num+ind_j+change[lb][l]] = Tlm[k*(2*lb+1)+l];
	    T[(ind_j+change[lb][l])*basis_num+ind_i+change[la][k]] = Tlm[k*(2*lb+1)+l];
	  }
	}
      }
      ind_j+=2*lb+1;
    }
    ind_i+=2*la+1;
  }
}
