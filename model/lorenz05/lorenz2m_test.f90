program lorenz2m_test

  use kind_module
  use lorenz2_module, only: lorenz2m_type

  implicit none
  type(lorenz2m_type) :: l2 
  integer :: nx
  integer, allocatable :: nks(:)
  real(kind=dp) :: dt, F 

  integer :: nt, nt6h, nsave, k, i
  real(kind=sp), allocatable :: xsave(:,:)
  character(len=100) :: fname

  nx = 240
  allocate( nks(4) )
  nks(1) = 8
  nks(2) = 16
  nks(3) = 32
  nks(4) = 64
  F = 15.0d0
  dt = 0.05d0 / 36.0d0

  nt6h = int(0.05d0 / dt)
  nt = 100 * 4 * nt6h
  nsave = nt / nt6h
  allocate( xsave(nsave+1,nx) )

  call l2%init(nx,nks,dt,F)
  l2%x(:) = F 
  l2%x(nx/2) = 1.001d0*F 
  xsave(1,:) = real(l2%x,kind=sp) 
  do k=1,nt
    print '(f5.2,a)', real(k-1,kind=dp)*100.0d0/real(nt,kind=dp),'%'
    call l2%run
    if(mod(k,nt6h)==0) then
      i=k/nt6h
      xsave(i+1,:) = real(l2%x,kind=sp)
    end if
  end do

  write(fname,'(a,i3,a,i1,4(a,i2),a)') &
  & "l05IIm_n",nx,"k",nks(1),"+",nks(2),"+",nks(3),"+",nks(4),&
  & "F",int(F),".grd"
  open(21,file=fname,access='direct',convert='big_endian',form='unformatted',recl=4*nx)
  do k=1,nsave+1
    write(21,rec=k) xsave(k,:)
  end do
  close(21)

end program lorenz2m_test