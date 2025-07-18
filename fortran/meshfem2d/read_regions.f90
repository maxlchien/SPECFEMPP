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

  subroutine read_regions()

! reads in material definitions in DATA/Par_file and outputs to num_material

  use constants, only: IMAIN,ANISOTROPIC_MATERIAL,POROELASTIC_MATERIAL, &
                       ELECTROMAGNETIC_MATERIAL,ISOTROPIC_COSSERAT_MATERIAL, &
                       TINYVAL,myrank

  use shared_parameters, only: nbregions,nbmodels,num_material,icodemat, &
    cp,cs, &
    rho_s_read,QKappa,Qmu, &
    aniso3,aniso4,aniso5,aniso6,aniso7,aniso8,aniso9,aniso10,aniso11, &
    rho_s ,kappa_s ,mu_s, nu_s, j_sc, lambda_sc, mu_sc, nu_sc, &
    nelmnts,nxread,nzread

  implicit none

  ! local parameters
  integer :: iregion,ix_start,ix_end,iz_start,iz_end,imaterial_number
  integer :: i,j,ielem,ier
  integer :: reread_nbregions
  double precision :: vpregion,vsregion,poisson_ratio
  logical :: is_overwriting
  integer :: id_already_set

  integer,external :: err_occurred

  ! safety check
  ! only main process is supposed to read in file parameters
  if (myrank /= 0) call stop_the_code('Only main process should read regions, exiting...')

  ! user output
  write(IMAIN,*) 'Regions:'
  write(IMAIN,*) '  Nb of regions in the mesh = ',nbregions
  write(IMAIN,*)

  ! assigns materials to mesh elements
  allocate(num_material(nelmnts),stat=ier)
  if (ier /= 0) call stop_the_code('Error allocating num_material array')
  num_material(:) = 0

  ! this call positions again the read header to the line with nbregions. we can then call next line to get the table
  call read_value_integer_p(reread_nbregions, 'mesher.nbregions')
  if (err_occurred() /= 0) call stop_the_code('Error reading parameter nbregions in Par_file')

  ! checks
  if (reread_nbregions /= nbregions) call stop_the_code('Error re-reading parameter nbregions in Par_file')
  if (nbmodels < 1) call stop_the_code('Invalid number of model definitions')

  ! read the material numbers for each region
  do iregion = 1,nbregions

    ! reads in region range
    ! format: #ix_start #ix_end #iz_start #iz_end #imaterial_number
    call read_region_coordinates_p(ix_start,ix_end,iz_start,iz_end,imaterial_number)

    ! check
    if (imaterial_number < 1) call stop_the_code('Negative material number not allowed!')
    if (ix_start < 1) call stop_the_code('Left coordinate of region negative!')
    if (ix_end > nxread) call stop_the_code('Right coordinate of region too high!')
    if (iz_start < 1) call stop_the_code('Bottom coordinate of region negative!')
    if (iz_end > nzread) call stop_the_code('Top coordinate of region too high!')

    if (iregion == 1) write(IMAIN,*) '------'
    write(IMAIN,*) 'Region ',iregion
    write(IMAIN,*) 'IX from ',ix_start,' to ',ix_end
    write(IMAIN,*) 'IZ from ',iz_start,' to ',iz_end

    ! note on supported model formats:
    !  acoustic                - model_number  1 rho    Vp   0   0   0 QKappa Qmu   0   0   0    0      0   0
    !  elastic                 - model_number  1 rho    Vp  Vs   0   0 QKappa Qmu   0   0   0    0      0   0
    !  anisotropic             - model_number  2 rho   c11 c13 c15 c33    c35 c55 c12 c23 c25    0 QKappa Qmu
    !  anisotropic (in AXISYM) - model_number  2 rho   c11 c13 c15 c33    c35 c55 c12 c23 c25  c22 QKappa Qmu
    !  poroelastic             - model_number  3 rhos rhof phi   c kxx    kxz kzz  Ks  Kf Kfr etaf   mufr Qmu
    !  electromagnetic         - model_number  4 mu0 e0 e11(e0) e33(e0) sig11 sig33 Qe11 Qe33 Qs11 Qs33 Qv 0 0
    ! isotropic cosserat       - model_number  5 rho kappa  mu nu j lambda_c mu_c nu_c  0   0    0      0   0
    !  tomo                    - model_number -1 0       0   A   0   0      0   0   0   0   0    0      0   0
    !
    ! in particular, icodemat(imaterial_number) can be negative for tomographic models

    ! determines region domain
    if (icodemat(imaterial_number) /= ANISOTROPIC_MATERIAL .and. icodemat(imaterial_number) /= POROELASTIC_MATERIAL &
         .and. icodemat(imaterial_number) /= ELECTROMAGNETIC_MATERIAL .and. &
         icodemat(imaterial_number) /= ISOTROPIC_COSSERAT_MATERIAL) then
      ! isotropic material
      vpregion = cp(imaterial_number)
      vsregion = cs(imaterial_number)

      write(IMAIN,*) 'Material # ',imaterial_number,' isotropic'
      if (vsregion < TINYVAL) then
        write(IMAIN,*) 'Material is fluid'
      else
        write(IMAIN,*) 'Material is solid'
      endif
      write(IMAIN,*) 'vp     = ',sngl(vpregion)
      write(IMAIN,*) 'vs     = ',sngl(vsregion)
      write(IMAIN,*) 'rho    = ',sngl(rho_s_read(imaterial_number))
      if (vpregion == vsregion) stop 'Materials cannot have Vs = Vp, there is an error in your input file'

      ! Poisson ratio (only for non-tomographic models, where vp/vs have been defined)
      if (icodemat(imaterial_number) > 0) then
        poisson_ratio = 0.5d0*(vpregion*vpregion - 2.d0*vsregion*vsregion) / (vpregion*vpregion - vsregion*vsregion)
        write(IMAIN,*) 'Poisson''s ratio = ',sngl(poisson_ratio)
        write(IMAIN,*) 'QKappa = ',sngl(QKappa(imaterial_number))
        write(IMAIN,*) 'Qmu    = ',sngl(Qmu(imaterial_number))

        if (poisson_ratio <= -1.00001d0 .or. poisson_ratio >= 0.50001d0) call stop_the_code('incorrect value of Poisson''s ratio')
      else
        ! tomographic material
        write(IMAIN,*) 'tomographic material = ',icodemat(imaterial_number)
      endif

    else if (icodemat(imaterial_number) == POROELASTIC_MATERIAL) then
      ! poroelastic material
      write(IMAIN,*) 'Material # ',imaterial_number,' isotropic'
      write(IMAIN,*) 'Material is poroelastic'

    else if (icodemat(imaterial_number) == ELECTROMAGNETIC_MATERIAL) then
      ! electromagentic material
      write(IMAIN,*) 'Material # ',imaterial_number,' electromagnetic'
      write(IMAIN,*) 'Material is electromagnetic'

    else if (icodemat(imaterial_number) == ISOTROPIC_COSSERAT_MATERIAL) then
      ! isotropic spin material
      write(IMAIN,*) 'Material # ',imaterial_number,' isotropic spin'
      write(IMAIN,*) 'Material is Elastic Isotropic Cosserat medium'
    else
      ! anisotropic material
      write(IMAIN,*) 'Material # ',imaterial_number,' anisotropic'
      write(IMAIN,*) 'cp = ',sngl(cp(imaterial_number))
      write(IMAIN,*) 'cs = ',sngl(cs(imaterial_number))
      write(IMAIN,*) 'c11 = ',sngl(aniso3(imaterial_number))
      write(IMAIN,*) 'c13 = ',sngl(aniso4(imaterial_number))
      write(IMAIN,*) 'c15 = ',sngl(aniso5(imaterial_number))
      write(IMAIN,*) 'c33 = ',sngl(aniso6(imaterial_number))
      write(IMAIN,*) 'c35 = ',sngl(aniso7(imaterial_number))
      write(IMAIN,*) 'c55 = ',sngl(aniso8(imaterial_number))
      write(IMAIN,*) 'c12 = ',sngl(aniso9(imaterial_number))
      write(IMAIN,*) 'c23 = ',sngl(aniso10(imaterial_number))
      write(IMAIN,*) 'c25 = ',sngl(aniso11(imaterial_number))
      write(IMAIN,*) 'rho = ',sngl(rho_s_read(imaterial_number))
      write(IMAIN,*) 'QKappa = ',sngl(QKappa(imaterial_number))
      write(IMAIN,*) 'Qmu = ',sngl(Qmu(imaterial_number))
    endif

    ! store density and velocity model
    is_overwriting = .false.
    do j = iz_start,iz_end
      do i = ix_start,ix_end
        ! element index
        ielem = (j-1)*nxread+i
        ! checks if element has been already assigned
        if (num_material(ielem) /= 0) then
          is_overwriting = .true.
          id_already_set = num_material(ielem)
        endif
        ! sets new material id for element
        num_material(ielem) = imaterial_number
      enddo
    enddo

    ! user output
    if (is_overwriting) then
      write(IMAIN,*) '*************************************'
      write(IMAIN,*) 'Warning: Element range from this region is overwriting material numbers previously set on elements.'
      write(IMAIN,*) '         This indicates that your region range is overlapping the region for material ',id_already_set
      write(IMAIN,*) '         If your regions should be exclusive, please fix the region definitions in the Par_file!'
      write(IMAIN,*) '*************************************'
    endif
    write(IMAIN,*) '------'
    call flush_IMAIN()

  enddo
  write(IMAIN,*)
  call flush_IMAIN()

  if (minval(num_material(:)) <= 0) call stop_the_code('Velocity model not entirely set...')

  end subroutine read_regions
