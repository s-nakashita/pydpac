program lorenz3m_test

  use kind_module
  use lorenz3_module, only: lorenz3m_type

  implicit none
  type(lorenz3m_type) :: l3 
  integer :: nx, ni
  integer, allocatable :: nks(:)
  real(kind=dp) :: b, c, dt, F

  integer :: nt, k, i, ksave, nsave
  real(kind=sp), allocatable :: zsave(:,:)
  character(len=100) :: fname
  character(len=100) :: finame
  real(kind=dp), allocatable :: z0(:)

  nx = 960
  allocate( nks(4) )
  nks(1) = 32
  nks(2) = 64
  nks(3) = 128
  nks(4) = 256
  ni = 12
  b = 10.0d0
  c = 0.6d0
  F = 15.0d0
  dt = 0.05d0 / 36.0d0 / b
  call l3%init(nx,nks,ni,b,c,dt,F)

  ksave = int(0.05d0/dt)
  nt = 100 * 4 * ksave
  nsave = nt / ksave
  allocate( zsave(nsave+1,nx) )

  allocate(z0(nx))
  write(finame,'(a,i3,2(a,i2),2(a,i3),2(a,i2),a,f4.1,a,f3.1,a)') &
  & "z0_n",nx,"k",nks(1),"+",nks(2),"+",nks(3),"+",nks(4),&
  & "i",ni,"F",int(F),"b",b,"c",c,".npy"
  open(11,file=finame,form='unformatted',access='stream')
  read(11) z0
  close(11)

  l3%z(:) = z0(:)
  zsave(1,:) = real(l3%z,kind=sp)
  do k=1,nt
    print '(f6.2,a)', real(k-1,kind=dp)*100.0d0/real(nt,kind=dp),'%'
    call l3%run
    if(mod(k,ksave)==0) then
      i=k/ksave
      zsave(i+1,:) = real(l3%z,kind=sp)
    end if
  end do

  write(fname,'(a,i3,2(a,i2),2(a,i3),2(a,i2),a,f4.1,a,f3.1,a)') &
  & "l05IIIm_n",nx,"k",nks(1),"+",nks(2),"+",nks(3),"+",nks(4),&
  & "i",ni,"F",int(F),"b",b,"c",c,".grd"
  open(21,file=fname,access='direct',convert='big_endian',form='unformatted',recl=4*nx)
  do k=1,nsave+1
    write(21,rec=k) zsave(k,:)
  end do
  close(21)

end program lorenz3m_test