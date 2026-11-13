! This file is part of the zen-opt fork of RESPACK, a modified version of RESPACK 20240804
! (original: https://sites.google.com/view/kazuma7k6r, Copyright (c) 2017 Kazuma Nakamura et al.).
! Copyright (C) 2026 Daniel Guterding
!
! This program is free software: you can redistribute it and/or modify it under the terms of the
! GNU General Public License as published by the Free Software Foundation, version 3 of the License.
! This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
! even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
! General Public License (GPL/GPLv3.txt) for more details.
!
! m_wfn_shared -- one copy of the single-precision wave functions per node (MPI-3 shared memory window).
! With one rank per q group (MPI_num_qcomm = number of ranks) every rank needs all bands at all irreducible k, and
! nobody modifies them after reading. The ranks of a node then share one copy: node rank 0 allocates the window
! (OCC4 and VIR4 back to back) and reads the file; the others only map it.
module m_wfn_shared
  use mpi
  use, intrinsic :: iso_c_binding
  implicit none
  private
  public :: wfn_shared_alloc, wfn_shared_sync, wfn_shared_free
  integer, save :: win = MPI_WIN_NULL, comm_node = MPI_COMM_NULL

contains

  ! Shared OCC4(NTG, nbo, nk), VIR4(NTG, nbv, nk) for the ranks of comm on the same node; leader = node rank 0
  ! (it reads and fills them). Returns .false. (nothing allocated) if the window cannot be created.
  logical function wfn_shared_alloc(comm, NTG, nbo, nbv, nk, occ, vir, leader) result(ok)
    integer, intent(in) :: comm, NTG, nbo, nbv, nk
    complex(4), pointer, contiguous, intent(out) :: occ(:, :, :), vir(:, :, :)
    logical, intent(out) :: leader
    integer(kind=MPI_ADDRESS_KIND) :: sz, qsz, baseptr
    integer :: ierr, du, nrank
    integer(8) :: no, nv
    type(c_ptr) :: base
    ok = .false.
    leader = .true.
    nullify(occ, vir)
    call MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm_node, ierr)
    if (ierr /= MPI_SUCCESS) return
    call MPI_Comm_rank(comm_node, nrank, ierr)
    leader = (nrank == 0)
    no = int(NTG, 8)*int(nbo, 8)*int(nk, 8)
    nv = int(NTG, 8)*int(nbv, 8)*int(nk, 8)
    sz = 0
    if (leader) sz = 8_MPI_ADDRESS_KIND*int(no + nv, MPI_ADDRESS_KIND)
    call MPI_Comm_set_errhandler(comm_node, MPI_ERRORS_RETURN, ierr)
    call MPI_Win_allocate_shared(sz, 8, MPI_INFO_NULL, comm_node, baseptr, win, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, ierr, 1, MPI_INTEGER, MPI_MAX, comm_node, du)
    if (ierr /= MPI_SUCCESS) then
      if (win /= MPI_WIN_NULL) call MPI_Win_free(win, du)
      call MPI_Comm_free(comm_node, du)
      leader = .true.
      return
    end if
    call MPI_Win_shared_query(win, 0, qsz, du, baseptr, ierr)
    base = transfer(baseptr, base)
    call c_f_pointer(base, occ, [NTG, nbo, nk])
    base = transfer(baseptr + 8_MPI_ADDRESS_KIND*int(no, MPI_ADDRESS_KIND), base)
    call c_f_pointer(base, vir, [NTG, nbv, nk])
    call MPI_Win_fence(0, win, ierr)                ! epoch for the leader's writes
    if (leader) then
      occ = (0.0, 0.0)
      vir = (0.0, 0.0)
    end if
    ok = .true.
  end function wfn_shared_alloc

  ! after the leader has filled the arrays: make them visible to all ranks of the node
  subroutine wfn_shared_sync()
    integer :: ierr
    if (win /= MPI_WIN_NULL) call MPI_Win_fence(0, win, ierr)
  end subroutine wfn_shared_sync

  subroutine wfn_shared_free()
    integer :: ierr
    if (win /= MPI_WIN_NULL) call MPI_Win_free(win, ierr)
    if (comm_node /= MPI_COMM_NULL) call MPI_Comm_free(comm_node, ierr)
  end subroutine wfn_shared_free

end module m_wfn_shared
