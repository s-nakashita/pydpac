module lorenz3_module

  use kind_module
  use lorenz2_module, only: advection
!$ use omp_lib

  implicit none
  private

  ! Lorenz III
  type lorenz3_type
    integer :: nx
    integer :: nk
    integer :: ni
    real(kind=dp) :: b
    real(kind=dp) :: c
    real(kind=dp) :: F 
    real(kind=dp) :: dt
    real(kind=dp), allocatable :: z(:)
    real(kind=dp), allocatable :: x(:) ! large-scale
    real(kind=dp), allocatable :: y(:) ! small-scale
    real(kind=dp), allocatable :: ztend(:) ! tendency
    real(kind=dp), allocatable :: filmat(:,:) ! spatial filter
    contains
      procedure :: init => lorenz3_init
      procedure :: tend => lorenz3_tend
      procedure :: run => lorenz3_run
  end type lorenz3_type

  ! Lorenz III with multiple advection
  type lorenz3m_type
    integer :: nx
    integer, pointer :: nks(:)
    integer :: ni
    real(kind=dp) :: b
    real(kind=dp) :: c
    real(kind=dp) :: F
    real(kind=dp) :: dt
    real(kind=dp), allocatable :: z(:)
    real(kind=dp), allocatable :: x(:) ! large-scale
    real(kind=dp), allocatable :: y(:) ! small-scale
    real(kind=dp), allocatable :: ztend(:) ! tendency
    real(kind=dp), allocatable :: filmat(:,:) ! spatial filter
    contains
      procedure :: init => lorenz3m_init
      procedure :: tend => lorenz3m_tend
      procedure :: run => lorenz3m_run
  end type lorenz3m_type

  public lorenz3_type, lorenz3m_type

contains
! Lorenz III
  subroutine lorenz3_init(l3,nx,nk,ni,b,c,dt,F)
    class(lorenz3_type) :: l3
    integer, intent(in) :: nx
    integer, intent(in) :: nk 
    integer, intent(in) :: ni
    real(kind=dp), intent(in) :: b
    real(kind=dp), intent(in) :: c
    real(kind=dp), intent(in) :: dt
    real(kind=dp), intent(in) :: F

    l3%nx = nx
    l3%nk = nk
    l3%ni = ni
    l3%b = b
    l3%c = c
    l3%dt = dt
    l3%F = F

    allocate(l3%z(nx))
    allocate(l3%x(nx),l3%y(nx))
    allocate(l3%ztend(nx))

    allocate(l3%filmat(nx,nx))
    call set_filter(l3%ni,l3%filmat)

    print '(3(a,i5),3(a,f5.2),a,es9.3)', &
    & 'nx=',l3%nx,' nk=',l3%nk,' ni=',l3%ni,&
    & ' b=',l3%b,' c=',l3%c,' F=',l3%F,' dt=',l3%dt
  end subroutine lorenz3_init

  subroutine lorenz3_tend(l3,z)
    class(lorenz3_type) :: l3
    real(kind=dp), intent(in) :: z(:)
    real(kind=dp), allocatable :: xadv(:), yadv(:), xyadv(:)
    integer :: i, i1, i2, i3

    allocate(xadv(l3%nx),yadv(l3%nx),xyadv(l3%nx))
    call decomp(l3%nx,z,l3%filmat,l3%x,l3%y)
    call advection(l3%nk, l3%x, xadv)
    yadv(:) = 0.0d0
    do i=1,l3%nx
      i1 = i-2
      i2 = i-1
      i3 = i+1
      if(i1<=0) then
        i1=i1+l3%nx
      elseif(i1>l3%nx) then
        i1=i1-l3%nx
      end if
      if(i2<=0) then
        i2=i2+l3%nx
      elseif(i2>l3%nx) then
        i2=i2-l3%nx
      end if
      if(i3<=0) then
        i3=i3+l3%nx
      elseif(i3>l3%nx) then
        i3=i3-l3%nx
      end if
      yadv(i) = (-1.0d0*l3%y(i1)*l3%y(i2)) + (l3%y(i2)*l3%y(i3))
    end do
    xyadv(:) = 0.0d0
    do i=1,l3%nx
      i1 = i-2
      i2 = i-1
      i3 = i+1
      if(i1<=0) then
        i1=i1+l3%nx
      elseif(i1>l3%nx) then
        i1=i1-l3%nx
      end if
      if(i2<=0) then
        i2=i2+l3%nx
      elseif(i2>l3%nx) then
        i2=i2-l3%nx
      end if
      if(i3<=0) then
        i3=i3+l3%nx
      elseif(i3>l3%nx) then
        i3=i3-l3%nx
      end if
      xyadv(i) = (-1.0d0*l3%y(i1)*l3%x(i2)) + (l3%y(i2)*l3%x(i3))
    end do
    l3%ztend = xadv + (l3%b*l3%b*yadv) + (l3%c*xyadv) &
    & - l3%x - (l3%b*l3%y) + l3%F

  end subroutine lorenz3_tend

  ! 4th order Runge-Kutta
  subroutine lorenz3_run(l3)
    class(lorenz3_type) :: l3
    real(kind=dp), dimension(:), allocatable :: k1,k2,k3,k4,ztmp

    allocate(k1(l3%nx),k2(l3%nx),k3(l3%nx),k4(l3%nx),ztmp(l3%nx))
    call l3%tend(l3%z)
    k1 = l3%ztend * l3%dt
    ztmp = l3%z + (0.5d0*k1)
    call l3%tend(ztmp)
    k2 = l3%ztend * l3%dt
    ztmp = l3%z + (0.5d0*k2)
    call l3%tend(ztmp)
    k3 = l3%ztend * l3%dt
    ztmp = l3%z + k3
    call l3%tend(ztmp)
    k4 = l3%ztend * l3%dt
    
    l3%z = l3%z + ((0.5d0*k1) + k2 + k3 + (0.5d0*k4))/3.0d0
    deallocate(k1,k2,k3,k4,ztmp)

  end subroutine lorenz3_run

! Lorenz III with multiple advection terms
  subroutine lorenz3m_init(l3,nx,nks,ni,b,c,dt,F)
    class(lorenz3m_type) :: l3
    integer, intent(in) :: nx
    integer, intent(in), target :: nks(:)
    integer, intent(in) :: ni
    real(kind=dp), intent(in) :: b
    real(kind=dp), intent(in) :: c
    real(kind=dp), intent(in) :: dt
    real(kind=dp), intent(in) :: F

    l3%nx = nx
    l3%nks => nks
    l3%ni = ni
    l3%b = b
    l3%c = c
    l3%dt = dt
    l3%F = F

    allocate(l3%z(nx))
    allocate(l3%x(nx),l3%y(nx))
    allocate(l3%ztend(nx))

    allocate(l3%filmat(nx,nx))
    call set_filter(l3%ni,l3%filmat)

    print '(2(a,i5),3(a,f5.2),a,es9.3)', &
    & 'nx=',l3%nx,' ni=',l3%ni,&
    & ' b=',l3%b,' c=',l3%c,' F=',l3%F,' dt=',l3%dt
    print *, 'nks=',l3%nks
  end subroutine lorenz3m_init

  subroutine lorenz3m_tend(l3,z)
    class(lorenz3m_type) :: l3
    real(kind=dp), intent(in) :: z(:)
    real(kind=dp), allocatable :: xadv(:), xadvtmp(:), yadv(:), xyadv(:)
    integer :: k, i, i1, i2, i3

    allocate(xadv(l3%nx),yadv(l3%nx),xyadv(l3%nx))
    allocate(xadvtmp(l3%nx))
    call decomp(l3%nx,z,l3%filmat,l3%x,l3%y)
    xadv = 0.0d0
    do k=1,size(l3%nks)
      call advection(l3%nks(k), l3%x, xadvtmp)
      xadv = xadv + xadvtmp
    end do
    yadv(:) = 0.0d0
    do i=1,l3%nx
      i1 = i-2
      i2 = i-1
      i3 = i+1
      if(i1<=0) then
        i1=i1+l3%nx
      elseif(i1>l3%nx) then
        i1=i1-l3%nx
      end if
      if(i2<=0) then
        i2=i2+l3%nx
      elseif(i2>l3%nx) then
        i2=i2-l3%nx
      end if
      if(i3<=0) then
        i3=i3+l3%nx
      elseif(i3>l3%nx) then
        i3=i3-l3%nx
      end if
      yadv(i) = (-1.0d0*l3%y(i1)*l3%y(i2)) + (l3%y(i2)*l3%y(i3))
    end do
    xyadv(:) = 0.0d0
    do i=1,l3%nx
      i1 = i-2
      i2 = i-1
      i3 = i+1
      if(i1<=0) then
        i1=i1+l3%nx
      elseif(i1>l3%nx) then
        i1=i1-l3%nx
      end if
      if(i2<=0) then
        i2=i2+l3%nx
      elseif(i2>l3%nx) then
        i2=i2-l3%nx
      end if
      if(i3<=0) then
        i3=i3+l3%nx
      elseif(i3>l3%nx) then
        i3=i3-l3%nx
      end if
      xyadv(i) = (-1.0d0*l3%y(i1)*l3%x(i2)) + (l3%y(i2)*l3%x(i3))
    end do
    l3%ztend = xadv + (l3%b*l3%b*yadv) + (l3%c*xyadv) &
    & - l3%x - (l3%b*l3%y) + l3%F

  end subroutine lorenz3m_tend

  ! 4th order Runge-Kutta
  subroutine lorenz3m_run(l3)
    class(lorenz3m_type) :: l3
    real(kind=dp), dimension(:), allocatable :: k1,k2,k3,k4,ztmp

    allocate(k1(l3%nx),k2(l3%nx),k3(l3%nx),k4(l3%nx),ztmp(l3%nx))
    call l3%tend(l3%z)
    k1 = l3%ztend * l3%dt
    ztmp = l3%z + (0.5d0*k1)
    call l3%tend(ztmp)
    k2 = l3%ztend * l3%dt
    ztmp = l3%z + (0.5d0*k2)
    call l3%tend(ztmp)
    k3 = l3%ztend * l3%dt
    ztmp = l3%z + k3
    call l3%tend(ztmp)
    k4 = l3%ztend * l3%dt
    
    l3%z = l3%z + ((0.5d0*k1) + k2 + k3 + (0.5d0*k4))/3.0d0
    deallocate(k1,k2,k3,k4,ztmp)

  end subroutine lorenz3m_run

  subroutine set_filter(ni,filmat)
    implicit none
    integer, intent(in) :: ni
    real(kind=dp), intent(inout) :: filmat(:,:)
    integer :: nifil, njfil
    real(kind=dp) :: fi2,fi3,fi4
    real(kind=dp) :: al, be, tmp
    integer :: i, j, js, je, jj

    nifil = size(filmat,1)
    njfil = size(filmat,2)
    fi2 = real(ni*ni,kind=dp)
    fi3 = fi2*real(ni,kind=dp)
    fi4 = fi3*real(ni,kind=dp)
    al = ((3.0d0*fi2)+3.0d0)/((2.0d0*fi3)+(4.0d0*real(ni,kind=dp)))
    be = ((2.0d0*fi2)+1.0d0)/(fi4+(2.0d0*fi2))
    do i=1,nifil
      js = i - ni
      je = i + ni
      if (js<=0) js=js+njfil
      if (je>njfil) je=je-njfil
      do j=1,njfil
        tmp = 0.0d0
        jj = j
        if (js<je) then
          if (j>=js .and. j<=je) then
            tmp = al - be*abs(jj-i)
          end if
        else
          if (j<=je .or. j>=js) then
            tmp = al - be*min(abs(jj-i),njfil-abs(jj-i))
          end if
        end if
        if (j==js .or. j==je) then
          tmp=tmp*0.5d0
        end if
        filmat(i,j) = tmp
      end do
    end do

  end subroutine set_filter

  subroutine decomp(nx,z,filmat,x,y)
    implicit none
    integer, intent(in) :: nx
    real(kind=dp), intent(in) :: z(:)
    real(kind=dp), intent(in) :: filmat(:,:)
    real(kind=dp), intent(out) :: x(:)
    real(kind=dp), intent(out) :: y(:)
    integer :: i,j

    x(:) = 0.0d0
    do j=1,nx
      do i=1,nx
        x(i) = x(i) + (filmat(i,j)*z(j))
      end do
    end do
    y(:) = z(:) - x(:)

  end subroutine decomp

end module lorenz3_module