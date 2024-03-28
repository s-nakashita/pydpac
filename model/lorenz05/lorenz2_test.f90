program lorenz2_test

  use kind_module
  use lorenz2_module, only: lorenz2_type

  implicit none
  type(lorenz2_type) :: l2 
  integer :: nx, nk
  real(kind=dp) :: dt, F

  integer :: nt, k
  real(kind=dp) :: tmax

  nx = 240
  nk = 8
  dt = 0.05
  F = 15.0
  tmax = 200.0
  nt = int(tmax/dt) + 1

  call l2%init(nx,nk,dt,F)
  l2%x(:) = F 
  l2%x(nx/2) = F*1.001d0
  print *, 'initial'
  print *, l2%x
  do k=1,nt 
    call l2%run
    if(k.eq.1) then
      print *, 1
      print *, l2%x
    end if
  end do
  print *, 'final'
  print *, l2%x 

end program lorenz2_test