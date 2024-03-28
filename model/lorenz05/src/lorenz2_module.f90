module lorenz2_module

  use kind_module

  implicit none
  private

  ! Lorenz II
  type lorenz2_type
    integer :: nx
    integer :: nk
    real(kind=dp) :: dt
    real(kind=dp) :: F
    real(kind=dp), allocatable :: x(:)
    real(kind=dp), allocatable :: xtend(:)
    contains
      procedure :: init => lorenz2_init
      procedure :: tend => lorenz2_tend
      procedure :: run => lorenz2_run
  end type lorenz2_type

  ! Lorenz II with multiple advection terms
  type lorenz2m_type
    integer :: nx
    integer, pointer :: nks(:)
    real(kind=dp) :: dt
    real(kind=dp) :: F
    real(kind=dp), allocatable :: x(:)
    real(kind=dp), allocatable :: xtend(:)
    contains
      procedure :: init => lorenz2m_init
      procedure :: tend => lorenz2m_tend
      procedure :: run => lorenz2m_run
  end type lorenz2m_type

  public lorenz2_type, lorenz2m_type, advection

contains
! Lorenz II
  subroutine lorenz2_init(l2,nx,nk,dt,F)
    class(lorenz2_type) :: l2
    integer, intent(in) :: nx
    integer, intent(in) :: nk
    real(kind=dp), intent(in) :: dt
    real(kind=dp), intent(in) :: F

    l2%nx = nx
    l2%nk = nk
    l2%dt = dt
    l2%F  = F

    allocate(l2%x(nx))
    allocate(l2%xtend(nx))

    print '(a,i5,a,i5,a,f5.2,a,es9.3)', &
    & "nx=",l2%nx," nk=",l2%nk," F=",l2%F," dt=",l2%dt
  end subroutine lorenz2_init

  subroutine lorenz2_tend(l2,x)
    class(lorenz2_type) :: l2
    real(kind=dp), intent(in) :: x(:)
    real(kind=dp), allocatable :: xadv(:)

    allocate(xadv(l2%nx))
    call advection(l2%nk,x,xadv)
    l2%xtend = xadv - x + l2%F
    deallocate(xadv)

  end subroutine lorenz2_tend

  ! 4th order Runge-Kutta
  subroutine lorenz2_run(l2)
    class(lorenz2_type) :: l2
    real(kind=dp), dimension(:), allocatable :: k1,k2,k3,k4,xtmp

    allocate(k1(l2%nx),k2(l2%nx),k3(l2%nx),k4(l2%nx),xtmp(l2%nx))
    call l2%tend(l2%x)
    k1 = l2%xtend * l2%dt
    xtmp = l2%x + 0.5d0*k1
    call l2%tend(xtmp)
    k2 = l2%xtend * l2%dt
    xtmp = l2%x + 0.5d0*k2
    call l2%tend(xtmp)
    k3 = l2%xtend * l2%dt
    xtmp = l2%x + k3
    call l2%tend(xtmp)
    k4 = l2%xtend * l2%dt
    
    l2%x = l2%x + (0.5d0*k1 + k2 + k3 + 0.5d0*k4)/3.0d0
    deallocate(k1,k2,k3,k4,xtmp)

  end subroutine lorenz2_run

! Lorenz II with multiple advection terms
  subroutine lorenz2m_init(l2,nx,nks,dt,F)
    class(lorenz2m_type) :: l2
    integer, intent(in) :: nx
    integer, intent(in), target :: nks(:)
    real(kind=dp), intent(in) :: dt
    real(kind=dp), intent(in) :: F

    l2%nx = nx
    l2%nks => nks
    l2%dt = dt
    l2%F  = F

    allocate(l2%x(nx))
    allocate(l2%xtend(nx))

    print '(a,i5,a,f5.2,a,es9.3)', &
    & "nx=",l2%nx," F=",l2%F," dt=",l2%dt
    print *, "nks=",l2%nks
  end subroutine lorenz2m_init

  subroutine lorenz2m_tend(l2,x)
    class(lorenz2m_type) :: l2
    real(kind=dp), intent(in) :: x(:)
    real(kind=dp), allocatable :: xadv(:), xadvtmp(:)

    integer :: nk,k

    allocate(xadv(l2%nx),xadvtmp(l2%nx))

    xadv = 0.0d0
    do k=1,size(l2%nks)
      nk = l2%nks(k)
      call advection(nk,x,xadvtmp)
      xadv = xadv + xadvtmp
    end do
    l2%xtend = xadv - x + l2%F
    deallocate(xadv,xadvtmp)

  end subroutine lorenz2m_tend

  subroutine lorenz2m_run(l2)
    class(lorenz2m_type) :: l2
    real(kind=dp), dimension(:), allocatable :: k1,k2,k3,k4,xtmp

    allocate(k1(l2%nx),k2(l2%nx),k3(l2%nx),k4(l2%nx),xtmp(l2%nx))
    call l2%tend(l2%x)
    k1 = l2%xtend * l2%dt
    xtmp = l2%x + 0.5d0*k1
    call l2%tend(xtmp)
    k2 = l2%xtend * l2%dt
    xtmp = l2%x + 0.5d0*k2
    call l2%tend(xtmp)
    k3 = l2%xtend * l2%dt
    xtmp = l2%x + k3
    call l2%tend(xtmp)
    k4 = l2%xtend * l2%dt
    
    l2%x = l2%x + (0.5d0*k1 + k2 + k3 + 0.5d0*k4)/3.0d0
    deallocate(k1,k2,k3,k4,xtmp)

  end subroutine lorenz2m_run

! calculation of [X,Y]_K term
  subroutine advection(nk,x,xadv,x2)
    implicit none
    integer, intent(in) :: nk
    real(kind=dp), intent(in) :: x(:)
    real(kind=dp), intent(out) :: xadv(:)
    real(kind=dp), intent(in), optional :: x2(:)

    integer :: n, nj, i, j, ii, jj
    real(kind=dp), allocatable :: w(:), w2(:), y(:)
    logical :: sumdiff

    if(mod(nk,2)==0) then
      sumdiff = .true.
      nj = nk / 2
    else
      sumdiff = .false.
      nj = (nk-1)/2
    end if

    n = size(x)
    allocate( w(n) )
    w(:) = 0.0d0
    do j=-nj,nj
      do i=1,n
        jj = i+j
        if(jj.le.0) then
          jj=jj+n
        elseif(jj.gt.n) then
          jj=jj-n
        end if
        w(i) = w(i) + x(jj)
      end do
    end do
    if(sumdiff) then
      do i=1,n
        jj = i+nj
        if(jj.le.0) then
          jj=jj+n
        elseif(jj.gt.n) then
          jj=jj-n 
        end if
        ii = i-nj
        if(ii.le.0) then
          ii=ii+n
        elseif(ii.gt.n) then
          ii=ii-n
        end if
        w(i) = w(i) - 0.5*x(jj) - 0.5*x(ii)
      end do
    end if
    w = w / float(nk)

    allocate( w2(n), y(n) )
    w2(:) = 0.0
    y(:) = 0.0
    if(present(x2)) then
      y(:) = x2(:)
      do j=-nj,nj
        do i=1,n
          jj = i+j
          if(jj.le.0) then
            jj=jj+n
          elseif(jj.gt.n) then
            jj=jj-n
          end if
          w2(i) = w2(i) + y(jj)
        end do
      end do
      if(sumdiff) then
        do i=1,n
          jj = i+nj
          if(jj.le.0) then
            jj=jj+n
          elseif(jj.gt.n) then
            jj=jj-n 
          end if
          ii = i-nj
          if(ii.le.0) then
            ii=ii+n
          elseif(ii.gt.n) then
            ii=ii-n
          end if
          w2(i) = w2(i) - 0.5*y(jj) - 0.5*y(ii)
        end do
      end if
      w2 = w2 / float(nk)
    else
      y(:) = x(:)
      w2(:) = w(:) 
    end if
    
    xadv(:) = 0.0d0
    do j=-nj,nj
      do i=1,n 
        jj=i-(nk-j)
        if(jj.le.0) then
          jj=jj+n
        elseif(jj.gt.n) then
          jj=jj-n 
        end if
        ii=i+nk+j
        if(ii.le.0) then
          ii=ii+n 
        elseif(ii.gt.n) then
          ii=ii-n
        end if
        xadv(i) = xadv(i) + w(jj)*y(ii)
      end do
    end do
    if(sumdiff) then
      do i=1,n
        jj=i-(nk+nj)
        if(jj.le.0) then
          jj=jj+n 
        elseif(jj.gt.n) then
          jj=jj-n 
        end if
        ii=i+nk-nj
        if(ii.le.0) then
          ii=ii+n
        elseif(ii.gt.n) then
          ii=ii-n
        end if
        xadv(i) = xadv(i) - 0.5*w(jj)*y(ii)
        jj=i-(nk-nj)
        if(jj.le.0) then
          jj=jj+n 
        elseif(jj.gt.n) then
          jj=jj-n 
        end if
        ii=i+nk+nj
        if(ii.le.0) then
          ii=ii+n
        elseif(ii.gt.n) then
          ii=ii-n
        end if
        xadv(i) = xadv(i) - 0.5*w(jj)*y(ii)
      end do
    end if
    xadv = xadv / float(nk)
    do i=1,n 
      jj=i-2*nk
      if(jj.le.0) then
        jj=jj+n
      elseif(jj.gt.n) then
        jj=jj-n
      end if
      ii=i-nk
      if(ii.le.0) then
        ii=ii+n 
      elseif(ii.gt.n) then
        ii=ii-n 
      end if
      xadv(i) = xadv(i) - w(jj)*w2(ii)
    end do
    return
  end subroutine advection

end module lorenz2_module