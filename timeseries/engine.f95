!FILE: ENGINE_NEW.F

!-----------------------------------------------------------
!--------- F95/03/08 re-write of unweighted scargle by -----
!--------- Cuypers / Aerts / De Cat / Degroote -------------
!--------- Modified by Johnston ----------------------------
!-----------------------------------------------------------

    subroutine scargle (x,y,f0,df,f1,s1,n,nf)

!  Computation of the scargle periodogram without the explicit
!  tau calculation, with iteration ( Method Cuypers)
!  Modified by Degroote, adapted by Johnston

!  These parameters deal with input / output to / from the code

    implicit real*8 (A-H,O-Z)
    integer n
    integer nf
    real(8) f0,df
    dimension x(n), y(n)
    dimension f1(nf), s1(nf)
    data twopi, dtwo, dnul /6.28318530717959D0,2.0D0,0.0D0/

!f2py intent(in) n
!f2py intent(in) nf
!f2py intent(in) x
!f2py intent(in) y
!f2py intent(in) f0
!f2py intent(in) df
!f2py intent(in) f1
!f2py intent(in) s1
!f2py depend(n) x
!f2py depend(nf) s1
!f2py intent(out) f1
!f2py intent(out) s1

!  Applying the depend(n/nf), we let f2py know that n and nf
!  are the sizes of the arrays x/nf, making them an optional parameter

!  Everything declared below this point is no longer input / output

    real(8), dimension(nf) :: ss(nf), sc(nf), ss2(nf), sc2(nf)
    real(8) :: f,twopi_f, twopi_df, dp_n, dp_nsq
    real(8) :: a_f0,s_f0,c_f0,s2_f0,c2_f0
    real(8) :: a_df,s_df,c_df,s2_df,c2_df
    real(8) :: a,y_i
    real(8) :: s_f0_yi, c_f0_yi, c_f0_yi_last, c2_f0_yi_last
    real(8) :: sck_sq, ssk_sq, sck2_sq, ssk2_sq
    integer :: i,j


    f = f0
    twopi_f = twopi * f
    twopi_df = twopi * df
    dp_n = DFLOAT(n)
    dp_nsq = dp_n*dp_n
    print *,dp_nsq

    do i=1,nf
      ss(i) = dnul
      sc(i) = dnul
      ss2(i) = dnul
      sc2(i) = dnul
    end do

    do i=1,n
      a = x(i)
      a_f0 = DMOD(a*twopi_f,twopi)
      s_f0 = DSIN(a_f0)
      c_f0 = DCOS(a_f0)
      s2_f0 = dtwo*s_f0*c_f0
      c2_f0 = c_f0**2 - s_f0**2

      a_df = DMOD(a*twoPi_df,TWOPI)
      s_df = DSIN(a_df)
      c_df = DCOS(a_df)
      s2_df = DTWO * s_df * c_df
      c2_df = c_df**2 - s_df**2
      y_i = y(i)

      c_f0_yi = c_f0 * y_i
      s_f0_yi = s_f0 * y_i

      do j=1,NF
        ss(j) = ss(j) + s_f0_yi
        sc(j) = sc(j) + c_f0_yi

        c_f0_yi_last = c_f0_yi
        c_f0_yi = c_f0_yi_last * c_df - s_f0_yi * s_df
        s_f0_yi = s_f0_yi * c_df + c_f0_yi_last * s_df

        ss2(j) = ss2(j) + s2_f0
        sc2(j) = sc2(j) + c2_f0

        c2_f0_yi_last = c2_f0
        c2_f0 = c2_f0_yi_last * c2_df - s2_f0 * s2_df
        s2_f0 = s2_f0 * c2_df + c2_f0_yi_last * s2_df

      enddo
    enddo

    do i=1,nf
      ssk_sq = ss(i)*ss(i)
      ss2k_sq = ss2(i)*ss2(i)
      sck_sq = sc(i)*sc(i)
      sc2k_sq = sc2(i)*sc2(i)

      f1(i) = f
      s1(i) = ( sck_sq*(dp_n-sc2(i)) + ssk_sq*(dp_n+sc2(i)) - &
                dtwo*ss(i)*sc(i)*ss2(i))/(dp_nsq-sc2k_sq-ss2k_sq)
      f = f+df
      print *,s1(i)
    enddo

    RETURN
    END

    program dummy
    end
