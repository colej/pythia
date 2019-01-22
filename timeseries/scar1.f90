SUBROUTINE scar1 ! Subroutine for fast calculation of the Lomb-Scargle periodogram (search for first frequency f1)

use light_curve
use lomb_scargle
IMPLICIT REAL(8) (A-H,O-Z)

DATA TWOPI, DTWO, DNUL/6.28318530717959D0,2.0D0,0.0D0/

F = FrequencyResolution
TPF = TWOPI * F
TDF = TWOPI * FrequencyStep
TN = real(ntimes,8)
TNSQ = TN * TN

do K = 1, NumberFrequencies
 SS(K) = DNUL
 SC(K) = DNUL
 SS2(K) = DNUL
 SC2(K) = DNUL
enddo

do I = 1, ntimes
 A = times(I)
 AF0 = DMOD(A*TPF,TWOPI)
 S0 = DSIN(AF0)
 S0N(I) = S0
 C0 = DCOS(AF0)
 C0N(I) = C0
 S20 = weights(i) * DTWO * S0 * C0
 C20 = weights(i) * (C0 * C0 - S0 * S0)

 ADF = DMOD(A*TDF,TWOPI)
 SDF = DSIN(ADF)
 SDFN(I) = SDF
 CDF = DCOS(ADF)
 CDFN(I) = CDF
 S2DF = DTWO * SDF * CDF
 C2DF = CDF * CDF - SDF * SDF
 XI = flux(I) * weights(i)
 C0X = C0 * XI
 S0X = S0 * XI

 do K = 1, NumberFrequencies
  SS(K) = SS(K) + S0X
  SC(K) = SC(K) + C0X
  CTX = C0X
  C0X = CTX * CDF - S0X * SDF
  S0X = S0X * CDF + CTX * SDF
  SS2(K) = SS2(K) + S20
  SC2(K) = SC2(K) + C20
  C2T = C20
  C20 = C2T * C2DF - S20 * S2DF
  S20 = S20 * C2DF + C2T * S2DF
 enddo

enddo

do K = 1, NumberFrequencies
 SSK = SS(K)
 SS2K = SS2(K)
 SCK = SC(K)
 SC2K = SC2(K)

 F1(K) = F
 PS(K) = (SCK*SCK*(TN-SC2K)+SSK*SSK*(TN+SC2K)-DTWO*SSK*SCK*SS2K) / (TNSQ-SC2K*SC2K-SS2K*SS2K)
 DEN(K) = (TNSQ-SC2K*SC2K-SS2K*SS2K)
 F = F + FrequencyStep
enddo

end
