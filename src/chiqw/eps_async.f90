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
! m_eps_async -- post-processing of one q point (q /= 0) of calc_chiqw: eps(G,G',w) = 1 - v chi0 and its
! inverse for all frequencies (OpenMP over the frequencies, single-threaded LAPACK each, as in chiqw.F),
! and the output files of dir-eps/qNNN (dat.sq, dat.epsqw.NNNNNN, dat.epsqw.NNN, dat.log.NNN, same names
! and contents as the original code).
!
! With flg_async_eps=1 this runs in a background thread while the main thread already computes chi0 of the
! next q point (useful when the GPU computes chi0 and the CPU cores are idle). The background thread makes
! no MPI calls and uses explicit paths instead of CHDIR, which is process-wide; the main thread waits for
! it (eps_post_wait) before any CHDIR or the next post-processing. chi0 is taken over without a copy
! (move_alloc) and overwritten with eps^-1, so no extra arrays of size NG*NG*nen are allocated.
module m_eps_async
  use iso_c_binding
  use omp_lib
  implicit none
  private
  public :: eps_post_q, eps_post_wait

  complex(8), allocatable, save :: chi(:, :, :)
  integer, allocatable, save :: lg(:, :)
  integer, save :: NG = 0, nen = 0, NTG = 0, qnum = 0, f_eps_base = 0, f_eps = 0, f_log = 0
  real(8), save :: qv(3), bb1(3), bb2(3), bb3(3), tpi, volume
  logical, save :: running = .false.

  interface
    integer(c_int) function respack_thread_start(f) bind(C, name='respack_thread_start')
      import :: c_int, c_funptr
      type(c_funptr), value :: f
    end function respack_thread_start
    integer(c_int) function respack_thread_join() bind(C, name='respack_thread_join')
      import :: c_int
    end function respack_thread_join
  end interface

contains

  ! Post-processing of q point qnum_ (dir-eps/qNNN). chiq (NG_, NG_, nen_), summed over the MPI ranks and
  ! not yet divided by the volume, is taken over and deallocated on return.
  subroutine eps_post_q(chiq, NTG_, NG_, nen_, LG0q, q1, q2, q3, b1, b2, b3, tpi_, volume_, qnum_, &
                        fbase, feps, flog, async)
    complex(8), allocatable, intent(inout) :: chiq(:, :, :)
    integer, intent(in) :: NTG_, NG_, nen_, LG0q(3, NTG_), qnum_, fbase, feps, flog
    real(8), intent(in) :: q1, q2, q3, b1(3), b2(3), b3(3), tpi_, volume_
    logical, intent(in) :: async

    call eps_post_wait()
    call move_alloc(chiq, chi)
    if (allocated(lg)) deallocate(lg)
    allocate(lg(3, NTG_))
    lg = LG0q
    NTG = NTG_; NG = NG_; nen = nen_; qnum = qnum_
    f_eps_base = fbase; f_eps = feps; f_log = flog
    qv = [q1, q2, q3]; bb1 = b1; bb2 = b2; bb3 = b3; tpi = tpi_; volume = volume_
    if (async) then
      running = .true.
      if (respack_thread_start(c_funloc(post_worker)) /= 0) then
        running = .false.
        write(6, *) 'eps_post_q: background thread not started; post-processing synchronously'
        call post_run()
      end if
    else
      call post_run()
    end if
  end subroutine eps_post_q

  ! Wait for a running background post-processing (no-op otherwise).
  subroutine eps_post_wait()
    integer(c_int) :: rc
    if (running) then
      rc = respack_thread_join()
      running = .false.
      if (rc /= 0) write(6, *) 'eps_post_wait: pthread_join returned', rc
    end if
  end subroutine eps_post_wait

  subroutine post_worker() bind(C)
    call post_run()
  end subroutine post_worker

  subroutine post_run()
    character(len=32) :: dir
    character(len=80) :: fname
    integer :: ie, u, ig, jg
    integer, allocatable :: fu(:)
    complex(8), allocatable :: epsbuf(:, :)
    real(8) :: t0, t1, t2

    t0 = omp_get_wtime()
    write(dir, "('dir-eps/q',i3.3,'/')") qnum
    chi = chi/volume

    open(newunit=u, file=trim(dir)//'dat.sq')
    write(u, *) qv(1), qv(2), qv(3)
    close(u)

    ! diagonal text files, one per frequency (calc_eps_rpa writes them)
    allocate(fu(nen))
    do ie = 1, nen
      write(fname, '("dat.epsqw.",i6.6)') f_eps_base + ie
      open(newunit=fu(ie), file=trim(dir)//trim(fname))
      write(fu(ie), '(a)') '#epsqw diagonal term'
      write(fu(ie), '(a)') '#1:q+G [au], 2:Re(eps-1), 3:Im(eps-1)'
    end do

    t1 = omp_get_wtime()
    !$omp parallel private(epsbuf)
    allocate(epsbuf(NG, NG))
    !$omp do schedule(dynamic, 1)
    do ie = 1, nen
      call calc_eps_rpa(NTG, NG, lg, qv(1), qv(2), qv(3), bb1, bb2, bb3, chi(1, 1, ie), tpi, epsbuf, 0, fu(ie))
      chi(:, :, ie) = epsbuf
      close(fu(ie))
    end do
    !$omp end do
    deallocate(epsbuf)
    !$omp end parallel
    t2 = omp_get_wtime()

    write(fname, '("dat.epsqw.",i3.3)') f_eps
    open(newunit=u, file=trim(dir)//trim(fname), form='unformatted')
    write(u) (((chi(ig, jg, ie), ig=1, NG), jg=1, NG), ie=1, nen)
    close(u)

    write(fname, '("dat.log.",i3.3)') f_log
    open(newunit=u, file=trim(dir)//trim(fname))
    write(u, *) '#'
    write(u, *) '#finish chi0 evaluation for', qnum
    write(u, *) '#'
    close(u)

    write(6, '(a,i4,a,3f9.2)') ' fork: q', qnum, ': time (s) prep / eps inversion / write eps', &
         t1 - t0, t2 - t1, omp_get_wtime() - t2
    flush(6)
    deallocate(chi, fu)
  end subroutine post_run

end module m_eps_async
