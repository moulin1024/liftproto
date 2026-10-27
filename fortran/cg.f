C ======================================================================
C     PCGLIB -- JACOBI-PRECONDITIONED CONJUGATE GRADIENT FOR
C               -DIV( K GRAD U ) = F  ON THE UNIT SQUARE, U=0 ON BOUNDARY
C     DELIBERATELY WRITTEN IN LEGACY FORTRAN 77 STYLE:
C       IMPLICIT TYPING, COMMON BLOCKS, WORK-ARRAY PARTITIONING,
C       GOTO CONTROL FLOW, MANUALLY UNROLLED BLAS-STYLE KERNELS.
C ======================================================================
      SUBROUTINE SETUP(NXI,NYI)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXN=4096)
      COMMON /GRID/ NX,NY,HX2I,HY2I
      COMMON /GEOM/ CK(MAXN),DG(MAXN)
      PI = 4.0D0*DATAN(1.0D0)
      NX = NXI
      NY = NYI
      HX = 1.0D0/DBLE(NX+1)
      HY = 1.0D0/DBLE(NY+1)
      HX2I = 1.0D0/(HX*HX)
      HY2I = 1.0D0/(HY*HY)
      DO 20 J=1,NY
      DO 10 I=1,NX
      K = (J-1)*NX+I
      CK(K) = 1.0D0 + 0.5D0*DSIN(2.0D0*PI*I*HX)*DCOS(PI*J*HY)
   10 CONTINUE
   20 CONTINUE
      DO 40 J=1,NY
      DO 30 I=1,NX
      K = (J-1)*NX+I
      CW = CK(K)
      CE = CK(K)
      CS = CK(K)
      CN = CK(K)
      IF (I.GT.1)  CW = CK(K-1)
      IF (I.LT.NX) CE = CK(K+1)
      IF (J.GT.1)  CS = CK(K-NX)
      IF (J.LT.NY) CN = CK(K+NX)
      DG(K) = HX2I*0.5D0*(CK(K)+CW) + HX2I*0.5D0*(CK(K)+CE)
     $      + HY2I*0.5D0*(CK(K)+CS) + HY2I*0.5D0*(CK(K)+CN)
   30 CONTINUE
   40 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
C     Y = A*X   (MATRIX FREE, LINEARIZED INDEX, GOTO-SKIPPED BOUNDARIES)
C ----------------------------------------------------------------------
      SUBROUTINE AMUL(N,X,Y)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXN=4096)
      DIMENSION X(*),Y(*)
      COMMON /GRID/ NX,NY,HX2I,HY2I
      COMMON /GEOM/ CK(MAXN),DG(MAXN)
      DO 20 J=1,NY
      DO 10 I=1,NX
      K = (J-1)*NX + I
      S = DG(K)*X(K)
      IF (I.EQ.1) GOTO 11
      S = S - HX2I*0.5D0*(CK(K)+CK(K-1))*X(K-1)
   11 IF (I.EQ.NX) GOTO 12
      S = S - HX2I*0.5D0*(CK(K)+CK(K+1))*X(K+1)
   12 IF (J.EQ.1) GOTO 13
      S = S - HY2I*0.5D0*(CK(K)+CK(K-NX))*X(K-NX)
   13 IF (J.EQ.NY) GOTO 14
      S = S - HY2I*0.5D0*(CK(K)+CK(K+NX))*X(K+NX)
   14 Y(K) = S
   10 CONTINUE
   20 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
C     Z = M**(-1) R   (JACOBI)
C ----------------------------------------------------------------------
      SUBROUTINE PCOND(N,R,Z)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MAXN=4096)
      DIMENSION R(*),Z(*)
      COMMON /GEOM/ CK(MAXN),DG(MAXN)
      DO 10 K=1,N
      Z(K) = R(K)/DG(K)
   10 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
C     BLAS-STYLE DOT PRODUCT, UNROLLED BY 5
C ----------------------------------------------------------------------
      DOUBLE PRECISION FUNCTION DDOT(N,DX,DY)
      DOUBLE PRECISION DX(*),DY(*),DTEMP
      INTEGER I,M,MP1,N
      DTEMP = 0.0D0
      DDOT = 0.0D0
      IF (N.LE.0) RETURN
      M = MOD(N,5)
      IF (M.EQ.0) GO TO 40
      DO 30 I = 1,M
         DTEMP = DTEMP + DX(I)*DY(I)
   30 CONTINUE
      IF (N.LT.5) GO TO 60
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
         DTEMP = DTEMP + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     $           DX(I+2)*DY(I+2) + DX(I+3)*DY(I+3) + DX(I+4)*DY(I+4)
   50 CONTINUE
   60 DDOT = DTEMP
      RETURN
      END
C ----------------------------------------------------------------------
C     BLAS-STYLE Y = Y + DA*X, UNROLLED BY 4
C ----------------------------------------------------------------------
      SUBROUTINE DAXPY(N,DA,DX,DY)
      DOUBLE PRECISION DA,DX(*),DY(*)
      INTEGER I,M,MP1,N
      IF (N.LE.0) RETURN
      IF (DA.EQ.0.0D0) RETURN
      M = MOD(N,4)
      IF (M.EQ.0) GO TO 40
      DO 30 I = 1,M
         DY(I) = DY(I) + DA*DX(I)
   30 CONTINUE
      IF (N.LT.4) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,4
         DY(I) = DY(I) + DA*DX(I)
         DY(I+1) = DY(I+1) + DA*DX(I+1)
         DY(I+2) = DY(I+2) + DA*DX(I+2)
         DY(I+3) = DY(I+3) + DA*DX(I+3)
   50 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
      SUBROUTINE DCOPY(N,DX,DY)
      DOUBLE PRECISION DX(*),DY(*)
      DO 10 I=1,N
      DY(I) = DX(I)
   10 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
C     P = Z + BETA*P
C ----------------------------------------------------------------------
      SUBROUTINE XPAY(N,Z,BETA,P)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Z(*),P(*)
      DO 10 I=1,N
      P(I) = Z(I) + BETA*P(I)
   10 CONTINUE
      RETURN
      END
C ----------------------------------------------------------------------
C     PCG DRIVER.  W IS A WORK ARRAY OF LENGTH 4*N, PARTITIONED BY HAND.
C ----------------------------------------------------------------------
      SUBROUTINE CGSOLV(N,B,X,W,TOL,MAXIT,ITER,RNORM)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(*),X(*),W(*)
      IR = 1
      IZ = IR + N
      IP = IZ + N
      IQ = IP + N
      CALL AMUL(N,X,W(IQ))
      DO 10 I=1,N
      W(IR+I-1) = B(I) - W(IQ+I-1)
   10 CONTINUE
      CALL PCOND(N,W(IR),W(IZ))
      CALL DCOPY(N,W(IZ),W(IP))
      RHO = DDOT(N,W(IR),W(IZ))
      BNRM = DSQRT(DDOT(N,B,B))
      ITER = 0
  100 CONTINUE
      RNORM = DSQRT(DDOT(N,W(IR),W(IR)))
      IF (RNORM .LE. TOL*BNRM) GOTO 900
      IF (ITER .GE. MAXIT) GOTO 900
      ITER = ITER + 1
      CALL AMUL(N,W(IP),W(IQ))
      ALPHA = RHO / DDOT(N,W(IP),W(IQ))
      CALL DAXPY(N,ALPHA,W(IP),X)
      CALL DAXPY(N,-ALPHA,W(IQ),W(IR))
      CALL PCOND(N,W(IR),W(IZ))
      RHONEW = DDOT(N,W(IR),W(IZ))
      BETA = RHONEW / RHO
      RHO = RHONEW
      CALL XPAY(N,W(IZ),BETA,W(IP))
      GOTO 100
  900 CONTINUE
      RETURN
      END
