program lorenz2_test

  use kind_module
  use lorenz2_module, only: lorenz2_type

  implicit none
  type(lorenz2_type) :: l2 
  integer :: nx, nk
  real(kind=dp) :: dt, F

  integer :: nt, k
  real(kind=dp) :: tmax
  real(kind=sp), allocatable :: xsave(:,:)
  character(len=100) :: fname

  nx = 240
  nk = 8
  dt = 0.05
  F = 15.0
  tmax = 20.0
  nt = int(tmax/dt)
  allocate(xsave(nt+1,nx))

  call l2%init(nx,nk,dt,F)
  l2%x(:) = F 
  l2%x(nx/2) = F*1.001d0
  print *, 'initial'
  print *, l2%x
  xsave(1,:) = real(l2%x,kind=sp)
  do k=1,nt 
    call l2%run
    xsave(k+1,:) = real(l2%x,kind=sp)
    if(k.eq.1) then
      print *, 1
      print *, l2%x
    end if
  end do
  print *, 'final'
  print *, l2%x 

  write(fname,'(a,i3,a,i1,a,i2,a)') &
  & "l05II_n",nx,"k",nk,"F",int(F),".grd"
  open(21,file=fname,access='direct',form='unformatted',recl=4*nx)
  do k=1, size(xsave,1)
    write(21,rec=k) xsave(k,:)
  end do
  close(21)

end program lorenz2_test