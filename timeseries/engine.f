C FILE: PYSCARGLE.F

C ---------------------------------------------------------------------------------------
C ----- Routine for weighted scarlge -- Cuypers / Aerts / De Cat
C ---------------------------------------------------------------------------------------

      SUBROUTINE SCAR2 (N,X,T,F0,NF,DF,F1,S1,SS,SC,SS2,SC2)

C     Computation of Scargles periodogram without explicit tau
C     calculation, with iteration (Method Cuypers)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(N),T(N)
      DIMENSION F1(NF),S1(NF)
      DIMENSION SS(NF),SC(NF),SS2(NF),SC2(NF)
      DATA TWOPI,DTWO,DNUL/6.28318530717959D0,2.0D0,0.0D0/
Cf2py intent(in) N
Cf2py intent(in) X
Cf2py intent(in) T
Cf2py intent(in) F0
Cf2py intent(in) NF
Cf2py intent(in) DF
Cf2py intent(in) F1
Cf2py intent(in) S1
Cf2py intent(in) SS
Cf2py intent(in) SC
Cf2py intent(in) SS2
Cf2py intent(in) SC2
Cf2py intent(out) S1
Cf2py intent(out) F1

      F = F0
      TPF = TWOPI * F
      TDF = TWOPI * DF
      TN = DFLOAT(N)
      TNSQ=TN*TN

      DO 20 K=1,NF
         SS(K)  = DNUL
         SC(K)  = DNUL
         SS2(K) = DNUL
         SC2(K) = DNUL
20       CONTINUE

      DO 40 I=1,N

         A = T(I)
         AF0 = DMOD(A*TPF,TWOPI)
         S0 = DSIN(AF0)
         C0 = DCOS(AF0)
         S20 = DTWO * S0 * C0
         C20 = C0 * C0 - S0 * S0

         ADF = DMOD(A*TDF,TWOPI)
         SDF = DSIN(ADF)
         CDF = DCOS(ADF)
         S2DF = DTWO * SDF * CDF
         C2DF = CDF * CDF - SDF * SDF
         XI=X(I)
         C0X = C0 * XI
         S0X = S0 * XI
         DO 30 K=1,NF
            SS(K) = SS(K) + S0X
            SC(K) = SC(K) + C0X
            CTX=C0X
            C0X = CTX * CDF - S0X * SDF
            S0X = S0X * CDF + CTX * SDF
            SS2(K) = SS2(K) + S20
            SC2(K) = SC2(K) + C20
            C2T = C20
            C20 = C2T * C2DF - S20 * S2DF
            S20 = S20 * C2DF + C2T * S2DF
30          CONTINUE

40       CONTINUE

      DO 50 K=1,NF
         SSK  =  SS(K)
         SS2K = SS2(K)
         SCK  =  SC(K)
         SC2K = SC2(K)

         F1(K)=F
         S1(K)=(SCK*SCK*(TN-SC2K)+SSK*SSK*(TN+SC2K)-DTWO*SSK*SCK*SS2K)/
     &      (TNSQ-SC2K*SC2K-SS2K*SS2K)
         F=F+DF

50    CONTINUE

      RETURN
      END


C ---------------------------------------------------------------------------------------
C ----- Routine for weighted scarlge -- Cuypers / Aerts / De Cat
C ---------------------------------------------------------------------------------------


      SUBROUTINE SCAR3 (N,X,T,F0,NF,DF,F1,S1,SS,SC,SS2,SC2,W)

C     Computation of Scargles periodogram without explicit tau
C     calculation, with iteration (Method Cuypers)
C     Weighted version!
C     SC2 = R2, SS2 =
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(N),T(N)
      DIMENSION F1(NF),S1(NF)
      DIMENSION SS(NF),SC(NF),SS2(NF),SC2(NF), W(N)
      DATA TWOPI,DTWO,DNUL/6.28318530717959D0,2.0D0,0.0D0/
Cf2py intent(in) N
Cf2py intent(in) X
Cf2py intent(in) T
Cf2py intent(in) F0
Cf2py intent(in) NF
Cf2py intent(in) DF
Cf2py intent(in) F1
Cf2py intent(in) S1
Cf2py intent(in) SS
Cf2py intent(in) SC
Cf2py intent(in) SS2
Cf2py intent(in) SC2
Cf2py intent(in) W
Cf2py intent(out) S1
Cf2py intent(out) F1

      F = F0
      TPF = TWOPI * F
      TDF = TWOPI * DF
      TN = DFLOAT(N)
      TNSQ=TN*TN

      DO 70 K=1,NF
         SS(K)  = DNUL
         SC(K)  = DNUL
         SS2(K) = DNUL
         SC2(K) = DNUL
70       CONTINUE

      DO 72 I=1,N

         A = T(I)
         AF0 = DMOD(A*TPF,TWOPI)
         S0 = DSIN(AF0)
         C0 = DCOS(AF0)
         S20 = DTWO * S0 * C0
         C20 = C0 * C0 - S0 * S0

         ADF = DMOD(A*TDF,TWOPI)
         SDF = DSIN(ADF)
         CDF = DCOS(ADF)
         S2DF = DTWO * SDF * CDF
         C2DF = CDF * CDF - SDF * SDF
         XI=X(I)
         WI=W(I)
         C0X = C0 * XI
         S0X = S0 * XI
         DO 72 K=1,NF
            SS(K) = SS(K) +  WI * S0X
            SC(K) = SC(K) +  WI * C0X
            CTX=C0X
            C0X = CTX * CDF - S0X * SDF
            S0X = S0X * CDF + CTX * SDF
            SS2(K) = SS2(K) +  WI * S20
            SC2(K) = SC2(K) +  WI * C20
            C2T = C20
            C20 = C2T * C2DF - S20 * S2DF
            S20 = S20 * C2DF + C2T * S2DF
73          CONTINUE

72       CONTINUE

      DO 74 K=1,NF
         SSK  =  SS(K)
         SS2K = SS2(K)
         SCK  =  SC(K)
         SC2K = SC2(K)

         F1(K)=F
         S1(K)=(SCK*SCK*(TN-SC2K)+SSK*SSK*(TN+SC2K)-DTWO*SSK*SCK*SS2K)/
     &      (TNSQ-SC2K*SC2K-SS2K*SS2K)
         F=F+DF

74    CONTINUE

      RETURN
      END
C      /* program dummy
C      end */


C ---------------------------------------------------------------------------------------
C ----- Routine for FASPER --> taken from numerical recipies
C ---------------------------------------------------------------------------------------

      SUBROUTINE fasper(x,y,n,ofac,hifac,wk1,wk2,nwk,nout,jmax,prob)
Cf2py intent(in) x
Cf2py intent(in) y
Cf2py intent(in) n
Cf2py intent(in) ofac
Cf2py intent(in) hifac
Cf2py intent(in,out) wk1
Cf2py intent(in,out) wk2
Cf2py intent(in,out) nwk
Cf2py intent(in,out) nout
Cf2py intent(in,out) jmax
Cf2py intent(in,out) prob
C     Given n data points with abscissas x (which need not be equally spaced)
C     and ordinates y, and given a desired oversampling factor ofac (a typical
C     value being 4 or larger), this rouinte fills array wk1 with a sequence of
C     nout increasing frequencies (not angular frequencies) up to hifac times
C     the average Nyquist frequency, and fills array wk2 with the values of the
C     Lomb normalized perioodgram at those frequencies. The arrays x and y are
C     not altered. Nwk, the dimension of wk1 and wk2, must be large enough for
C     intermediate work space, or an error (puase) results. The routine also
C     returns jmax such that wk2(jmax) is the maximum element in wk2, and prob,
C     an estimate of the significance of that maximum against the hypothesis of
C     random noise. A small value of prob inidicates that a signifcant periodic
C     signal is present
      INTEGER jmax,n,nout,nwk,MACC
      DOUBLE PRECISION hifac,ofac,prob,wk1(nwk),wk2(nwk),x(n),y(n)
      PARAMETER (MACC=4)
CU    USES avevar,realft,spread
      INTEGER j,k,ndim,nfreq,nfreqt
      DOUBLE PRECISION ave,ck,ckk,cterm,cwt,den,df,effm,expy,fac,fndim,
     *hc2wt,hs2wt,hypo,pmax,sterm,swt,var,xdif,xmax,xmin
      nout=0.5*ofac*hifac*n
      nfreqt=ofac*hifac*n*MACC
      nfreq=64
1     if (nfreq.lt.nfreqt) then
        nfreq=nfreq*2
      goto 1
      endif
      ndim=2*nfreq
      if(ndim.gt.nwk) pause 'workspaces too small in fasper'
      call avevar(y,n,ave,var)
      xmin=x(1)
      xmax=xmin
      do 11 j=2,n
        if(x(j).lt.xmin)xmin=x(j)
        if(x(j).gt.xmax)xmax=x(j)
11    continue
      xdif=xmax-xmin
      do 12 j=1,ndim
        wk1(j)=0.
        wk2(j)=0.
12    continue
      fac=ndim/(xdif*ofac)
      fndim=ndim
      do 13 j=1,n
        ck=1.+mod((x(j)-xmin)*fac,fndim)
        ckk=1.+mod(2.*(ck-1.),fndim)
        call spread(y(j)-ave,wk1,ndim,ck,MACC)
        call spread(1.,wk2,ndim,ckk,MACC)
13    continue
      call realft(wk1,ndim,1)
      call realft(wk2,ndim,1)
      df=1./(xdif*ofac)
      k=3
      pmax=-1.
      do 14 j=1,nout
        hypo=sqrt(wk2(k)**2+wk2(k+1)**2)
        hc2wt=0.5*wk2(k)/hypo
        hs2wt=0.5*wk2(k+1)/hypo
        cwt=sqrt(0.5+hc2wt)
        swt=sign(sqrt(0.5-hc2wt),hs2wt)
        den=0.5*n+hc2wt*wk2(k)+hs2wt*wk2(k+1)
        cterm=(cwt*wk1(k)+swt*wk1(k+1))**2/den
        sterm=(cwt*wk1(k+1)-swt*wk1(k))**2/(n-den)
        wk1(j)=j*df
        wk2(j)=(cterm+sterm)/(2.*var)
        if (wk2(j).gt.pmax) then
          pmax=wk2(j)
          jmax=j
        endif
        k=k+2
14    continue
      expy=exp(-pmax)
      effm=2.*nout/ofac
      prob=effm*expy
      if(prob.gt.0.01)prob=1.-(1.-expy)**effm
      return
      END


      SUBROUTINE spread(y,yy,n,x,m)
      INTEGER m,n
      DOUBLE PRECISION x,y,yy(n)
      INTEGER ihi,ilo,ix,j,nden,nfac(10)
      DOUBLE PRECISION fac
      SAVE nfac
      DATA nfac /1,1,2,6,24,120,720,5040,40320,362880/
      if(m.gt.10) pause 'factorial table too small in spread'
      ix=x
      if(x.eq.float(ix))then
        yy(ix)=yy(ix)+y
      else
        ilo=min(max(int(x-0.5*m+1.0),1),n-m+1)
        ihi=ilo+m-1
        nden=nfac(m)
        fac=x-ilo
        do 11 j=ilo+1,ihi
          fac=fac*(x-j)
11      continue
        yy(ihi)=yy(ihi)+y*fac/(nden*(x-ihi))
        do 12 j=ihi-1,ilo,-1
          nden=(nden/(j+1-ilo))*(j-ihi)
          yy(j)=yy(j)+y*fac/(nden*(x-j))
12      continue
      endif
      return
      END


      SUBROUTINE avevar(data,n,ave,var)
      INTEGER n
      DOUBLE PRECISION ave,var,data(n)
      INTEGER j
      DOUBLE PRECISION s,ep
      ave=0.0
      do 11 j=1,n
        ave=ave+data(j)
11    continue
      ave=ave/n
      var=0.0
      ep=0.0
      do 12 j=1,n
        s=data(j)-ave
        ep=ep+s
        var=var+s*s
12    continue
      var=(var-ep**2/n)/(n-1)
      return
      END

      SUBROUTINE realft(data,n,isign)
      INTEGER isign,n
      DOUBLE PRECISION data(n)
CU    USES four1
      INTEGER i,i1,i2,i3,i4,n2p3
      DOUBLE PRECISION c1,c2,h1i,h1r,h2i,h2r,wis,wrs
      DOUBLE PRECISION theta,wi,wpi,wpr,wr,wtemp
      theta=3.141592653589793d0/dble(n/2)
      c1=0.5
      if (isign.eq.1) then
        c2=-0.5
        call four1(data,n/2,+1)
      else
        c2=0.5
        theta=-theta
      endif
      wpr=-2.0d0*sin(0.5d0*theta)**2
      wpi=sin(theta)
      wr=1.0d0+wpr
      wi=wpi
      n2p3=n+3
      do 11 i=2,n/4
        i1=2*i-1
        i2=i1+1
        i3=n2p3-i2
        i4=i3+1
        wrs=sngl(wr)
        wis=sngl(wi)
        h1r=c1*(data(i1)+data(i3))
        h1i=c1*(data(i2)-data(i4))
        h2r=-c2*(data(i2)+data(i4))
        h2i=c2*(data(i1)-data(i3))
        data(i1)=h1r+wrs*h2r-wis*h2i
        data(i2)=h1i+wrs*h2i+wis*h2r
        data(i3)=h1r-wrs*h2r+wis*h2i
        data(i4)=-h1i+wrs*h2i+wis*h2r
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
11    continue
      if (isign.eq.1) then
        h1r=data(1)
        data(1)=h1r+data(2)
        data(2)=h1r-data(2)
      else
        h1r=data(1)
        data(1)=c1*(h1r+data(2))
        data(2)=c1*(h1r-data(2))
        call four1(data,n/2,-1)
      endif
      return
      END


      SUBROUTINE four1(data,nn,isign)
      INTEGER isign,nn
      DOUBLE PRECISION data(2*nn)
      INTEGER i,istep,j,m,mmax,n
      DOUBLE PRECISION tempi,tempr
      DOUBLE PRECISION theta,wi,wpi,wpr,wr,wtemp
      n=2*nn
      j=1
      do 11 i=1,n,2
        if(j.gt.i)then
          tempr=data(j)
          tempi=data(j+1)
          data(j)=data(i)
          data(j+1)=data(i+1)
          data(i)=tempr
          data(i+1)=tempi
        endif
        m=n/2
1       if ((m.ge.2).and.(j.gt.m)) then
          j=j-m
          m=m/2
        goto 1
        endif
        j=j+m
11    continue
      mmax=2
2     if (n.gt.mmax) then
        istep=2*mmax
        theta=6.28318530717959d0/(isign*mmax)
        wpr=-2.d0*sin(0.5d0*theta)**2
        wpi=sin(theta)
        wr=1.d0
        wi=0.d0
        do 13 m=1,mmax,2
          do 12 i=m,n,istep
            j=i+mmax
            tempr=sngl(wr)*data(j)-sngl(wi)*data(j+1)
            tempi=sngl(wr)*data(j+1)+sngl(wi)*data(j)
            data(j)=data(i)-tempr
            data(j+1)=data(i+1)-tempi
            data(i)=data(i)+tempr
            data(i+1)=data(i+1)+tempi
12        continue
          wtemp=wr
          wr=wr*wpr-wi*wpi+wr
          wi=wi*wpr+wtemp*wpi+wi
13      continue
        mmax=istep
      goto 2
      endif
      return
      END
C      /* program dummy
C      end */


C ---------------------------------------------------------------------------------------
C ----- Routine for discrete fourier transform -- Cuypers / Aerts / De Cat
C ---------------------------------------------------------------------------------------
C FILE: PYDFT.F
      SUBROUTINE FT(XX,TSAM,NN,WZ,NFREQ,SI,LFREQ,T0,MM,DF,FTRX,FTIX,O,W)

	  DIMENSION XX(NN), TSAM(NN)
	  integer NN
	  real*8 WZ
	  integer NFREQ
      real*8 si
      integer LFREQ
      real*8 T0,DF
      integer MM
	  DIMENSION FTRX(MM), FTIX(MM), O(MM),W(MM)
	  REAL*8 WTAN, SSIN
	  COMPLEX WORK

Cf2py intent(in) XX
Cf2py intent(in) TSAM
Cf2py intent(in) NN
Cf2py intent(in) WZ
Cf2py intent(in) NFREQ
Cf2py intent(in) SI
Cf2py intent(in) LFREQ
Cf2py intent(in) T0
Cf2py intent(in) MM
Cf2py intent(in) DF
Cf2py intent(in) FTRX
Cf2py intent(in) FTIX
Cf2py intent(in) O
Cf2py intent(in) W
Cf2py intent(out) FTRX
Cf2py intent(out) FTIX
Cf2py intent(out) O
Cf2py intent(out) W

	  TOL1 = 1.0 E -04
	  TOL2 = 1.0 E -08
	  WUSE = WZ
	  FNN = FLOAT( NN )
	  CONST1 = 1.0 / SQRT(2.0)
	  CONST2 = SI * CONST1
	  SUMT = 0.0
	  SUMX = 0.0
	  DO 100 I=1,NN
	     SUMT = SUMT + TSAM( I )
		 SUMX = SUMX + XX( I )
100   CONTINUE
	  ISTOP = NFREQ




	  TAU0 = SUMT / FNN         ! LIMIT
	  CSUM = FNN
	  SSUM = 0.0
	  FTRX(1) = SUMX / SQRT( FNN )
	  FTIX(1) = 0.0
	  WDEL = DF
	  WRUN = WUSE
      O(1) = 0.
      W(1) = FTRX(1)
	  II = 2


150   CONTINUE
      CSUM = 0.0
	  SSUM = 0.0
	  SUMTC = 0.0
	  SUMTS = 0.0

	  DO 190 I = 1,NN
	     TTT = TSAM( I )
		 ARG1 = 2.0 * WRUN * TTT
		 ARG = FOLD( ARG1 )
		 TCOS = COS ( ARG )
		 TSIN = SIN ( ARG )
		 CSUM = CSUM + TCOS
		 SSUM = SSUM + TSIN
190   CONTINUE

	  WATAN = ATAN2( SSUM, CSUM )
	  IF(ABS(SSUM).GT.TOL1 .OR. ABS(CSUM).GT.TOL1) GOTO 200
	  WATAN = ATAN2( -SUMTC, SUMTS)
200   CONTINUE

	  WTAU = 0.5 * WTAN
	  WTNEW = WTAU
	  SUMR = 0.0
	  SUMI = 0.0
	  SCOS2 = 0.0
	  SSIN2 = 0.0
	  CROSS = 0.0

	  DO 440 I = 1,NN

	     TIM = TSAM(I)
		 ARG1 = WRUN * TIM - WTNEW
		 ARG = FOLD( ARG1 )
		 TCOS = COS(ARG)
		 TSIN = SIN(ARG)

		 CROSS = CROSS + TIM * TCOS * TSIN
		 SCOS2 = SCOS2 + TCOS * TCOS
		 SSIN2 = SSIN2 + TSIN * TSIN

		 XD = XX(I)
		 SUMR = SUMR + XD * TCOS
		 SUMI = SUMI + XD * TSIN

440      CONTINUE

	  FTRD = CONST1 * SUMR / SQRT(SCOS2)
	  IF ( SSIN .LE. TOL1 ) GOTO 450
	  FTID = CONST2 * SUMI / SQRT(SSIN2)
	  GOTO 460

450   CONTINUE
      FTID = CONST2 * SUMX / SQRT( FNN )
	  IF ( ABS(CROSS) .GT. TOL2 ) FTID = 0.0

460   CONTINUE

	  PHASE1 = WTNEW - WRUN * T0
	  PHASE = FOLD( PHASE1 )
	  WORK = CMPLX( FTRD, FTID)*CEXP( CMPLX(0.0, PHASE ) )
	  FTRX(II) = REAL( WORK )
	  FTIX(II) = AIMAG( WORK )
	  O(II) = WRUN
	  W(II) = WORK
	  II = II + 1
	  WRUN = WRUN + WDEL
	  IF( II .LE. ISTOP )GOTO 150




	  IF( 2 * NFREQ .GT. LFREQ ) GOTO 999
	  I1 = NFREQ + 1

	  DO 320 I= I1,LFREQ
	     FTRX(I) = 0.0
		 FTIX(I) = 0.0
320      CONTINUE

	  NSTOP = LFREQ / 2
	  DO 340 I=2, NSTOP
	     IPUT = LFREQ -I + 2
		 FTRX(IPUT) =  FTRX(I)
		 FTIX(IPUT) = -FTIX(I)
340      CONTINUE
465   CONTINUE

	  RETURN
999   CONTINUE
      END


	  FUNCTION FOLD( ARG )

	  PI = 3.1415926535898
	  ARGMAX = 8000.0 * PI
	  FOLD = ARG
10    CONTINUE
      IF( FOLD .LE. ARGMAX) GOTO 20
	     FOLD = FOLD - ARGMAX
		 GOTO 10
20    CONTINUE
      IF( FOLD .GT. -ARGMAX) GOTO 30
	     FOLD = FOLD + ARGMAX
		 GOTO 20
30    CONTINUE
      RETURN
	  END
      program dummy
      end
