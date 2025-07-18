!========================================================================
!
!                            S P E C F E M 2 D
!                            -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".
!
!========================================================================

  subroutine determine_acoustic_surface()

  use constants, only: ANISOTROPIC_MATERIAL,TINYVAL,IMAIN,myrank,ELECTROMAGNETIC_MATERIAL, &
    POROELASTIC_MATERIAL

  use part_unstruct_par, only: nelem_acoustic_surface,acoustic_surface, &
    nxread,nzread,elmnts

  use shared_parameters, only: AXISYM,NGNOD,num_material,icodemat,phi_read,xmin_param, &
    absorbbottom,absorbleft,absorbright,absorbtop

  implicit none

  ! local parameters
  integer :: i,j,ier
  integer :: imaterial_number

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '  determining acoustic free surface...'
    call flush_IMAIN()
  endif

  ! count the number of acoustic free-surface elements
  nelem_acoustic_surface = 0

  ! if the surface is absorbing, it cannot be free at the same time
  if (.not. absorbtop) then
    j = nzread
    do i = 1,nxread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
     (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
       endif
    enddo
  endif
  if (.not. absorbbottom) then
    j = 1
    do i = 1,nxread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
       endif
    enddo
  endif
  ! in the axisymmetric case if xmin == 0 the axis is a symmetry axis and thus cannot be a free surface as well
  if (.not. absorbleft .and. .not. (AXISYM .and. abs(xmin_param) < TINYVAL)) then
    i = 1
    do j = 1,nzread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
       endif
    enddo
  endif
  if (.not. absorbright) then
    i = nxread
    do j = 1,nzread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
       endif
    enddo
  endif

  ! allocates surface elements
  allocate(acoustic_surface(4,nelem_acoustic_surface),stat=ier)
  if (ier /= 0) call stop_the_code('Error allocating acoustic_surface array')
  acoustic_surface(:,:) = 0
  nelem_acoustic_surface = 0

  ! 'acoustic_surface' contains
  ! 1/ element number,
  ! 2/ number of nodes that form the free surface,
  ! 3/ first node on the free surface,
  ! 4/ second node on the free surface, if relevant (if 2/ is equal to 2)

  if (.not. absorbtop) then
    j = nzread
    do i = 1,nxread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
          acoustic_surface(1,nelem_acoustic_surface) = (j-1)*nxread + (i-1)
          acoustic_surface(2,nelem_acoustic_surface) = 2
          acoustic_surface(3,nelem_acoustic_surface) = elmnts(3+NGNOD*((j-1)*nxread+i-1))
          acoustic_surface(4,nelem_acoustic_surface) = elmnts(2+NGNOD*((j-1)*nxread+i-1))
       endif
    enddo
  endif
  if (.not. absorbbottom) then
    j = 1
    do i = 1,nxread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
          acoustic_surface(1,nelem_acoustic_surface) = (j-1)*nxread + (i-1)
          acoustic_surface(2,nelem_acoustic_surface) = 2
          acoustic_surface(3,nelem_acoustic_surface) = elmnts(0+NGNOD*((j-1)*nxread+i-1))
          acoustic_surface(4,nelem_acoustic_surface) = elmnts(1+NGNOD*((j-1)*nxread+i-1))
       endif
    enddo
  endif
  ! in the axisymmetric case if xmin == 0 the axis is a symmetry axis and thus cannot be a free surface as well
  if (.not. absorbleft .and. .not. (AXISYM .and. abs(xmin_param) < TINYVAL)) then
    i = 1
    do j = 1,nzread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
          acoustic_surface(1,nelem_acoustic_surface) = (j-1)*nxread + (i-1)
          acoustic_surface(2,nelem_acoustic_surface) = 2
          acoustic_surface(3,nelem_acoustic_surface) = elmnts(0+NGNOD*((j-1)*nxread+i-1))
          acoustic_surface(4,nelem_acoustic_surface) = elmnts(3+NGNOD*((j-1)*nxread+i-1))
       endif
    enddo
  endif
  if (.not. absorbright) then
    i = nxread
    do j = 1,nzread
       imaterial_number = num_material((j-1)*nxread+i)
       if ((icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL) .and. (phi_read(imaterial_number) >= 1.d0) .and. &
       (icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL) .and. (icodemat(imaterial_number) /= POROELASTIC_MATERIAL)) then
          nelem_acoustic_surface = nelem_acoustic_surface + 1
          acoustic_surface(1,nelem_acoustic_surface) = (j-1)*nxread + (i-1)
          acoustic_surface(2,nelem_acoustic_surface) = 2
          acoustic_surface(3,nelem_acoustic_surface) = elmnts(1+NGNOD*((j-1)*nxread+i-1))
          acoustic_surface(4,nelem_acoustic_surface) = elmnts(2+NGNOD*((j-1)*nxread+i-1))
       endif
    enddo
  endif

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) '  number of acoustic elements with free surface = ',nelem_acoustic_surface
    write(IMAIN,*)
    call flush_IMAIN()
  endif

  end subroutine determine_acoustic_surface
