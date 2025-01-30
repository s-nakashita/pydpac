program lorenz3_test

  use kind_module
  use lorenz3_module, only: lorenz3_type

  implicit none
  type(lorenz3_type) :: l3
  integer :: nx, nk, ni
  real(kind=dp) :: b, c, dt, F

  integer :: nt, k, i, ksave, nsave
  real(kind=dp) :: tmax
  real(kind=sp), allocatable :: zsave(:,:)
  character(len=100) :: fname
  character(len=100) :: finame
  real(kind=dp), allocatable :: z0(:)

  nx = 960
  nk = 32
  ni = 12
  b = 10.0d0
  c = 0.6d0
  F = 15.0d0
  dt = 0.05d0 / b
  call l3%init(nx,nk,ni,b,c,dt,F)
  
  tmax = 20.0d0
  nt = ceiling(tmax/dt)
  ksave = int(b)
  nsave = nt / ksave
  allocate(zsave(nsave+1,nx))

  allocate(z0(nx))
  write(finame,'(a,i3,3(a,i2),a,f3.1,a)') &
  & "z0_n",nx,"k",nk,"i",ni,"F",int(F),"c",c,".npy"
  open(11,file=finame,form='unformatted',access='stream')
  read(11) z0
  close(11)
  print *, z0

  l3%z(:) = z0(:)
  zsave(1,:) = real(l3%z,kind=sp)
  do k=1,nt
    call l3%run
    if(mod(k,ksave)==0) then
      i=k/ksave
      zsave(i+1,:) = real(l3%z,kind=sp)
    end if
    if(k.eq.1) then
      print *, 1
      print *, l3%z
    end if
  end do
  print *, 'final'
  print *, l3%z

  write(fname,'(a,i3,3(a,i2),a,f3.1,a)') &
  & "l05III_n",nx,"k",nk,"i",ni,"F",int(F),"c",c,".grd"
  open(21,file=fname,access='direct',convert='big_endian',form='unformatted',recl=4*nx)
  do k=1, nsave+1
    write(21,rec=k) zsave(k,:)
  end do
  close(21)

  open(21,file="filmat.grd",access='direct',convert='big_endian',form='unformatted',recl=4*nx)
  do k=1,nx
    write(21,rec=k) real(l3%filmat(:,k),kind=sp)
  end do
  close(21)

end program lorenz3_test