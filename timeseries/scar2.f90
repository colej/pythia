SUBROUTINE scar2 ! Subroutine for fast calculation of the Lomb-Scargle periodogram (search for other frequencies after f1 has been searched for, the solely time or frequency dependent parts need not be computed again, these are stored during the first search for f1)

use light_curve
use lomb_scargle
IMPLICIT REAL*8 (A-H,O-Z)

DATA DTWO, DNUL/2.0D0,0.0D0/

TN = real(ntimes,8)

do K = 1, NumberFrequencies
 SS(K) = DNUL
 SC(K) = DNUL
enddo

do I = 1, ntimes
 S0 = S0N(I)
 C0 = C0N(I)
 S20 = weights(i) * DTWO * S0 * C0
 C20 = weights(i) * C0 * C0 - S0 * S0

 SDF = SDFN(I)
 CDF = CDFN(I)
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
 enddo

enddo 

do K = 1, NumberFrequencies
 SSK = SS(K)
 SS2K = SS2(K)
 SCK = SC(K)
 SC2K = SC2(K)

 PS(K) = (SCK*SCK*(TN-SC2K)+SSK*SSK*(TN+SC2K)-DTWO*SSK*SCK*SS2K) / DEN(K)
enddo

end
