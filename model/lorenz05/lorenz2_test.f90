program lorenz2_test

  use kind_module
  use lorenz2_module, only: lorenz2_type

  implicit none
  type(lorenz2_type) :: l2 
  integer :: nx, nk
  real(kind=dp) :: dt, F

  integer :: nt, k
  real(kind=dp) :: tmax
  real(kind=dp), allocatable :: xsave(:,:)
  real(kind=dp), allocatable :: xtsave(:,:)
  real(kind=dp), allocatable :: x0(:)
  character(len=100) :: fname
  character(len=100) :: finame

  !nx = 240
  !nk = 8
  !F = 15.0d0
  nx = 40
  nk = 1
  F = 1.0
  dt = 0.05d0
  tmax = 20.0d0
  nt = ceiling(tmax/dt)
  allocate(xsave(nt+1,nx))
  allocate(xtsave(nt+1,nx))

  call l2%init(nx,nk,dt,F)
  !l2%x(:) = F 
  !l2%x(nx/2) = F*1.0001d0
  allocate( x0(nx) )
  !write(finame,'(a,i3,a,i1,a,i2,a)') &
  write(finame,'(a,i2,a,i1,a,i1,a)') &
  & "x0_n",nx,"k",nk,"F",int(F),".npy"
  open(11,file=finame,form='unformatted',access='stream')
  read(11) x0
  close(11)
  l2%x(:) = x0(:)
  
  print *, 'initial'
  print *, l2%x
  !xsave(1,:) = real(l2%x,kind=sp)
  xsave(1,:) = l2%x
  call l2%tend(l2%x)
  xtsave(1,:) = l2%xtend
  do k=1,nt 
    !call l2%run
    call l2%run_euler
    !xsave(k+1,:) = real(l2%x,kind=sp)
    xsave(k+1,:) = l2%x
    call l2%tend(l2%x)
    xtsave(k+1,:) = l2%xtend
    if(k.eq.1) then
      print *, 1
      print *, l2%x
    end if
  end do
  print *, 'final'
  print *, l2%x 

  !write(fname,'(a,i3,a,i1,a,i2,a)') &
  write(fname,'(a,i2,a,i1,a,i1,a)') &
  & "l05II_euler_n",nx,"k",nk,"F",int(F),".grd"
  !open(21,file=fname,access='direct',convert='big_endian',form='unformatted',recl=4*nx)
  !do k=1, nt+1
  !  write(21,rec=k) xsave(k,:)
  !end do
  open(21,file=fname,access='stream',form='unformatted')
  write(21) xsave
  close(21)
  write(fname,'(a,i2,a,i1,a,i1,a)') &
  & "l05II_euler_tend_n",nx,"k",nk,"F",int(F),".grd"
  open(21,file=fname,access='stream',form='unformatted')
  write(21) xtsave
  close(21)

end program lorenz2_test