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
