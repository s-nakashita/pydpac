module kind_module
  implicit none
  public

  integer, parameter :: dp=kind(0.d0)
  integer, parameter :: sp=kind(0.e0)

end module kind_module