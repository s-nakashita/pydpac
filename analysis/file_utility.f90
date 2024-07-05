module file_utility
    implicit none
contains
    subroutine file_open(unit, file)
        implicit none
        integer, intent(in) :: unit
        character(len=*), intent(in) :: file

        open(unit=unit, file=file, position="append")
        return
    end subroutine file_open

    subroutine file_close(unit)
        implicit none
        integer, intent(in) :: unit
        
        close(unit)
        return
    end subroutine file_close
end module file_utility
