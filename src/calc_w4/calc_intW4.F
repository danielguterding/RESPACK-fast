      PROGRAM CALC_W4 
!--
      use m_rdinput       
      use fft_3d 
      include"config.h" 
      call read_input 
!--
      WRITE(6,*) 
      WRITE(6,*)'========================================'
      WRITE(6,*)'===     CALC_W4 CALCULATION START    ==='
      WRITE(6,*)'===     1. CALC R=0 CASE             ==='
      WRITE(6,*)'===     2. CALC SINGLE FREQUENCY     ==='
      WRITE(6,*)'========================================'
      WRITE(6,*) 
!--
!OPEN(117,R,FILE='dat.bandcalc') 
      OPEN(117,FILE='./dir-wfn/dat.bandcalc') 
      rewind(117) 
      read(117,*) Ecut_for_psi 
      read(117,*) FermiEnergy  
      read(117,*) Etot
      write(6,*)'Ecut_for_psi=',Ecut_for_psi 
      write(6,*)'FermiEnergy=',FermiEnergy  
      write(6,*)'Etot=',Etot
!--
!OPEN(105,R,FILE='dat.lattice')
      OPEN(105,FILE='./dir-wfn/dat.lattice') 
      REWIND(105)
      READ(105,*) a1(1),a1(2),a1(3)!a1 vector
      READ(105,*) a2(1),a2(2),a2(3)!a2 vector
      READ(105,*) a3(1),a3(2),a3(3)!a3 vector
      write(6,*)'FINISH REDING a1,a2,a3'
      CLOSE(105)
      call OUTER_PRODUCT(a2(1),a3(1),b1(1))
      !VOLUME=a1(1)*b1(1)+a1(2)*b1(2)+a1(3)*b1(3)
      VOLUME=abs(a1(1)*b1(1)+a1(2)*b1(2)+a1(3)*b1(3)) !20240212 Kazuma Nakamura
      b1(:)=b1(:)*tpi/VOLUME 
      call OUTER_PRODUCT(a3(1),a1(1),b2(1))
      b2(:)=b2(:)*tpi/VOLUME 
      call OUTER_PRODUCT(a1(1),a2(1),b3(1))
      b3(:)=b3(:)*tpi/VOLUME 
      write(6,*) 
      write(6,*)'======================'
      write(6,*)'CHECK OF (A_VEC*B_VEC)'
      write(6,*)'======================'
      write(6,*) 
      write(6,*)'a1*b1=',b1(1)*a1(1)+b1(2)*a1(2)+b1(3)*a1(3)
      write(6,*)'a2*b2=',b2(1)*a2(1)+b2(2)*a2(2)+b2(3)*a2(3)
      write(6,*)'a3*b3=',b3(1)*a3(1)+b3(2)*a3(2)+b3(3)*a3(3)
      write(6,*)'VOLUME OF UNIT CELL=',VOLUME  
!--
!OPEN(100,R,FILE='dat.symmetry') 
      OPEN(100,FILE='./dir-wfn/dat.symmetry') 
      rewind(100) 
      read(100,*) nsymq 
      read(100,*) nnp
      allocate(rg(3,3,nsymq));rg=0
      allocate(pg(3,nsymq));pg=0
      allocate(rginv(3,3,nsymq));rginv(:,:,:)=0.0d0 
      do iop=1,nsymq
       read(100,*)((rg(i,j,iop),i=1,3),j=1,3) 
       read(100,*)(pg(i,iop),i=1,3)   
      enddo 
      CLOSE(100) 
      rginv=rg 
      do iop=1,nsymq
      call invmat(3,rginv(1,1,iop)) 
      enddo 
      write(6,*)'finish rg'
!check 
!do iop=1,nsymq
!write(6,*) iop
!do i=1,3
!write(6,'(3I5,1x,3F15.10)') (rg(i,j,iop),j=1,3),(rginv(i,j,iop),j=1,3)
!enddo 
!enddo 
!do iop=1,nsymq
!do i=1,3
!do j=1,3
!s=0.0d0 
!do k=1,3
!s=s+rg(i,k,iop)*rginv(k,j,iop)
!enddo 
!write(6,*) i,j,s
!enddo 
!enddo 
!enddo 
!--
!OPEN(101,R,FILE='dat.sample-k') 
      OPEN(101,FILE='./dir-wfn/dat.sample-k') 
      rewind(101) 
      read(101,*) Nk_irr 
      allocate(SKI(3,Nk_irr));SKI(:,:)=0.0D0 
      do ik=1,Nk_irr 
       read(101,*)(SKI(i,IK),i=1,3) 
      enddo 
      close(101) 
      write(6,*)'finish ski'
      call est_NTK(NK_irr,SKI(1,1),NTK,nkb1,nkb2,nkb3)  
      write(6,'(a24,4i10)')'nkb1,nkb2,nkb3,NTK=',nkb1,nkb2,nkb3,NTK  
!20170327 
      Na1=nkb1/2;Na2=nkb2/2;Na3=nkb3/2
      write(6,'(a24,3i10)')'Na1,Na2,Na3=',Na1,Na2,Na3 
!--
!OPEN(132,R,FILE='dat.nkm') 
      OPEN(132,FILE='./dir-wfn/dat.nkm') 
      allocate(NGI(Nk_irr));NGI(:)=0
      rewind(132)
      do ik=1,Nk_irr 
       read(132,*) NGI(ik) 
      enddo 
      close(132) 
      NTG=maxval(abs(NGI(:))) 
      write(6,'(a10,i10)')'NTG=',NTG  
!--
!OPEN(104,R,FILE='dat.kg') 
      OPEN(104,FILE='./dir-wfn/dat.kg') 
      rewind(104) 
      allocate(KGI(3,NTG,Nk_irr));KGI(:,:,:)=0 
      do ik=1,Nk_irr 
       read(104,*) NG_for_psi 
       NGI(IK)=NG_for_psi 
       do ig=1,NG_for_psi 
        read(104,*)(KGI(i,ig,ik),i=1,3) 
       enddo 
      enddo  
      close(104)
      write(6,*)'finish kgi'
      !--
      allocate(LKGI(NTG,Nk_irr));LKGI=0.0d0 
      do ik=1,Nk_irr 
       do ig=1,NGI(ik) 
       ktmp(1)=(SKI(1,ik)+dble(KGI(1,ig,ik)))*b1(1)
     +        +(SKI(2,ik)+dble(KGI(2,ig,ik)))*b2(1) 
     +        +(SKI(3,ik)+dble(KGI(3,ig,ik)))*b3(1) 

       ktmp(2)=(SKI(1,ik)+dble(KGI(1,ig,ik)))*b1(2)
     +        +(SKI(2,ik)+dble(KGI(2,ig,ik)))*b2(2) 
     +        +(SKI(3,ik)+dble(KGI(3,ig,ik)))*b3(2) 

       ktmp(3)=(SKI(1,ik)+dble(KGI(1,ig,ik)))*b1(3)
     +        +(SKI(2,ik)+dble(KGI(2,ig,ik)))*b2(3) 
     +        +(SKI(3,ik)+dble(KGI(3,ig,ik)))*b3(3) 
       LKGI(ig,ik)=ktmp(1)**2+ktmp(2)**2+ktmp(3)**2
       enddo!ig 
       write(6,*) maxval(LKGI(:,ik))
      enddo!ik 
      write(6,*) 
      write(6,*) maxval(LKGI(:,:))
      Ecut_for_psi=maxval(LKGI(:,:))+1.0d-8
      write(6,*)'Ecut_for_psi=',Ecut_for_psi 
      !--
      L1=maxval(abs(KGI(1,:,:)))+1; write(6,*)'L1=',L1 
      L2=maxval(abs(KGI(2,:,:)))+1; write(6,*)'L2=',L2 
      L3=maxval(abs(KGI(3,:,:)))+1; write(6,*)'L3=',L3 
      allocate(packing(-L1:L1,-L2:L2,-L3:L3,Nk_irr))   
      packing(:,:,:,:)=0 
      do ik=1,Nk_irr 
       do ig=1,NGI(ik) 
        i1=KGI(1,ig,ik);j1=KGI(2,ig,ik);k1=KGI(3,ig,ik) 
        packing(i1,j1,k1,ik)=ig 
       enddo 
      enddo 
      !--
      !fft grid 20180309 
      !--
      !
      !m1=maxval(abs(KGI(1,:,:))) 
      !m2=maxval(abs(KGI(2,:,:))) 
      !m3=maxval(abs(KGI(3,:,:))) 
      !nwx2=algn235(2*m1) 
      !nwy2=algn235(2*m2) 
      !nwz2=algn235(2*m3) 
      !
      htmp=dsqrt(dot_product(a1,a1))
      h1(:)=a1(:)/htmp 
      htmp=dsqrt(dot_product(a2,a2))
      h2(:)=a2(:)/htmp 
      htmp=dsqrt(dot_product(a3,a3)) 
      h3(:)=a3(:)/htmp 
      d1=abs(dot_product(b1,h1)) 
      d2=abs(dot_product(b2,h2)) 
      d3=abs(dot_product(b3,h3)) 
      qwf=2.0d0*dsqrt(Ecut_for_psi) 
      !
      nwx2=algn235(int(qwf/d1)+1)!20180412 by Y.Nomura 
      nwy2=algn235(int(qwf/d2)+1)!20180412 by Y.Nomura 
      nwz2=algn235(int(qwf/d3)+1)!20180412 by Y.Nomura 
      !
      nfft1=nwx2+1
      nfft2=nwy2+1
      nfft3=nwz2+1
      Nl123=nfft1*nfft2*nfft3 
      !
      write(6,*)'nwx2=',nwx2 
      write(6,*)'nwy2=',nwy2 
      write(6,*)'nwz2=',nwz2 
      write(6,*)'nfft1=',nfft1 
      write(6,*)'nfft2=',nfft2 
      write(6,*)'nfft3=',nfft3 
      write(6,*)'NL123=',Nl123  
      call fft3_init(nwx2,nwy2,nwz2,nfft1,nfft2,nfft3,fs) 
!--
!gen(SK0,numirr,numrot,trs,RW) 
      allocate(SK0(3,NTK));SK0(:,:)=0.0d0
      allocate(numirr(NTK));numirr(:)=0
      allocate(numrot(NTK));numrot(:)=0
      allocate(trs(NTK));trs(:)=0
      allocate(RW(3,NTK));RW(:,:)=0
!20161207 
!      do ik=1,Nk_irr 
!       SK0(:,ik)=SKI(:,ik) 
!       numirr(ik)=ik;numrot(ik)=1;trs(ik)=1;RW(1:3,ik)=0
!      enddo 
!      jk=Nk_irr 
!--
      jk=0
      do ik=1,Nk_irr
      do iop=1,Nsymq
      !--
      !sym
      !--
       ktmp(:)=0.0d0;RWtmp(:)=0  
       ktmp(1)=rg(1,1,iop)*SKI(1,ik)+rg(1,2,iop)*SKI(2,ik)
     +        +rg(1,3,iop)*SKI(3,ik)
       ktmp(2)=rg(2,1,iop)*SKI(1,ik)+rg(2,2,iop)*SKI(2,ik)
     +        +rg(2,3,iop)*SKI(3,ik)
       ktmp(3)=rg(3,1,iop)*SKI(1,ik)+rg(3,2,iop)*SKI(2,ik)
     +        +rg(3,3,iop)*SKI(3,ik)
       call kcheck(ktmp(1),RWtmp(1))!rewind check 
       do iik=1,jk
        if(abs(SK0(1,iik)-ktmp(1))<1.0d-4.and. 
     +     abs(SK0(2,iik)-ktmp(2))<1.0d-4.and. 
     +     abs(SK0(3,iik)-ktmp(3))<1.0d-4)goto 1000
       enddo!iik
       jk=jk+1
       SK0(:,jk)=ktmp(:)
       numirr(jk)=ik;numrot(jk)=iop;trs(jk)=1;RW(:,jk)=RWtmp(:)
!time-reversal
1000   ktmp(:)=0.0d0;RWtmp(:)=0  
       ktmp(1)=rg(1,1,iop)*SKI(1,ik)+rg(1,2,iop)*SKI(2,ik)
     +        +rg(1,3,iop)*SKI(3,ik)
       ktmp(2)=rg(2,1,iop)*SKI(1,ik)+rg(2,2,iop)*SKI(2,ik)
     +        +rg(2,3,iop)*SKI(3,ik) 
       ktmp(3)=rg(3,1,iop)*SKI(1,ik)+rg(3,2,iop)*SKI(2,ik)
     +        +rg(3,3,iop)*SKI(3,ik) 
       call kcheck_trs(ktmp(1),RWtmp(1))!rewind check 20170322
       do iik=1,jk
        if(abs(SK0(1,iik)-(-ktmp(1)))<1.0d-4.and. 
     +     abs(SK0(2,iik)-(-ktmp(2)))<1.0d-4.and. 
     +     abs(SK0(3,iik)-(-ktmp(3)))<1.0d-4)goto 2000
       enddo!iik
       jk=jk+1
       SK0(:,jk)=-ktmp(:)
       numirr(jk)=ik;numrot(jk)=iop;trs(jk)=-1;RW(:,jk)=RWtmp(:)
2000  enddo 
      enddo 
      !--
      if(NTK/=jk)then 
       write(6,*)'ERROR;STOP;NTK should be jk'   
       write(6,*)'NTK=',NTK,'jk=',jk;STOP
      endif 
      !--   
      write(6,*) 
      write(6,*)'====='
      write(6,*)' SK0 '
      write(6,*)'====='
      write(6,*) 
      DO ik=1,NTK
       WRITE(6,'(I5,3F12.5)') ik,SK0(:,ik) 
      ENDDO 
!--
!gen(NG0,KG0) 
      allocate(KGtmp(3,NTG));KGtmp(:,:)=0 
      allocate(KG0(3,NTG,NTK));KG0(:,:,:)=0 
      allocate(NG0(NTK));NG0(:)=0
!20161207 
!      do jk=1,Nk_irr 
!       NG0(jk)=NGI(jk);KG0(:,:,jk)=KGI(:,:,jk)
!      enddo 
!      do jk=Nk_irr+1,NTK 
!--
      do jk=1,NTK 
       if(trs(jk)==1)then 
        ik=numirr(jk);iop=numrot(jk) 
        ktmp(1)=rg(1,1,iop)*SKI(1,ik)+rg(1,2,iop)*SKI(2,ik)
     +         +rg(1,3,iop)*SKI(3,ik)+dble(RW(1,jk)) 
        ktmp(2)=rg(2,1,iop)*SKI(1,ik)+rg(2,2,iop)*SKI(2,ik)
     +         +rg(2,3,iop)*SKI(3,ik)+dble(RW(2,jk))  
        ktmp(3)=rg(3,1,iop)*SKI(1,ik)+rg(3,2,iop)*SKI(2,ik)
     +         +rg(3,3,iop)*SKI(3,ik)+dble(RW(3,jk))  
        call make_KG0(NTG,b1(1),b2(1),b3(1),Ecut_for_psi,
     +       ktmp(1),ktmp(2),ktmp(3),KG0(1,1,jk),NG_for_psi)
        if(NG_for_psi/=NGI(ik))then 
         write(6,*)'ERROR;STOP;NG_for_psi should be NGI(ik)'   
         write(6,*)'NG_for_psi=',NG_for_psi,'NGI(ik)=',NGI(ik);STOP
        endif 
        NG0(jk)=NG_for_psi  
       elseif(trs(jk)==-1)then  
        ik=numirr(jk); iop=numrot(jk) 
        ktmp(1)=rg(1,1,iop)*SKI(1,ik)+rg(1,2,iop)*SKI(2,ik)
     +         +rg(1,3,iop)*SKI(3,ik)+dble(RW(1,jk)) 
        ktmp(2)=rg(2,1,iop)*SKI(1,ik)+rg(2,2,iop)*SKI(2,ik)
     +         +rg(2,3,iop)*SKI(3,ik)+dble(RW(2,jk))  
        ktmp(3)=rg(3,1,iop)*SKI(1,ik)+rg(3,2,iop)*SKI(2,ik)
     +         +rg(3,3,iop)*SKI(3,ik)+dble(RW(3,jk))  
        KGtmp(:,:)=0 
        call make_KG0(NTG,b1(1),b2(1),b3(1),Ecut_for_psi,
     +       ktmp(1),ktmp(2),ktmp(3),KGtmp(1,1),NG_for_psi)
        if(NG_for_psi/=NGI(ik))then 
         write(6,*)'ERROR;STOP;NG_for_psi should be NGI(ik)'   
         write(6,*)'NG_for_psi=',NG_for_psi,'NGI(ik)=',NGI(ik);STOP
        endif 
        NG0(jk)=NG_for_psi  
        KG0(:,:,jk)=-KGtmp(:,:) 
       endif 
      enddo 
      !--
      write(6,*) 
      write(6,*)'====='
      write(6,*)' NG0 '
      write(6,*)'====='
      write(6,*) 
      do ik=1,NTK 
       write(6,'(1I10)') NG0(ik) 
      enddo 
!--
!OPEN(113,R,FILE='dat.wan',FORM='unformatted') 
      OPEN(113,FILE='./dir-wan/dat.wan',FORM='unformatted') 
      REWIND(113)       
      read(113) NWF 
      allocate(C0(NTG,NWF,NTK));C0(:,:,:)=0.0D0 
      do ik=1,NTK
       read(113)((C0(ig,iw,ik),ig=1,NG0(ik)),iw=1,NWF)           
      enddo
      write(6,*)'FINISH REDING C0'
      CLOSE(113)  
!--
!      do ik=1,Nk_irr 
!       do iw=1,NWF
!        do jw=1,NWF 
!         SUM_CMPX=0.0d0 
!         do ig=1,NG0(ik) 
!          SUM_CMPX=SUM_CMPX+CONJG(C0(ig,iw,ik))*C0(ig,jw,ik) 
!         enddo 
!         write(6,'(3i5,x,2f15.10)') ik,iw,jw,SUM_CMPX 
!        enddo 
!       enddo 
!      enddo 
!--
!OPEN(134,R,FILE='dat.wan-center') 
      OPEN(134,FILE='./dir-wan/dat.wan-center') 
      allocate(coord(3,NWF));coord(:,:)=0.0d0    
      rewind(134)
      read(134,'(a)') dum_char
      read(134,'(a)') dum_char
      do iw=1,NWF
       read(134,*)(coord(i,iw),i=1,3) 
      enddo 
!--
!OPEN(300,R,FILE='dat.chi_cutoff')!20170402  
       OPEN(300,FILE='./dir-eps/dat.chi_cutoff')!20170402 
       read(300,*) Ecut_for_eps!20170402  
!--
!OPEN(135,R,FILE='W-grid') 
      OPEN(135,FILE='./dir-eps/dat.wgrid') 
      rewind(135) 
      read(135,*) Num_freq_grid!20170402
      ne=Num_freq_grid!20170402
      allocate(em(ne));em(:)=0.0d0 
      do ie=1,ne 
       !read(135,*) em(ie) 
       read(135,'(2f15.10)') em(ie)!20171212
      enddo 
!--
!OPEN(400,R,FILE='dat.log.400')!20170420  
!OPEN(301,R,FILE='dat.sq')!20170420   
      Nq_irr=Nk_irr 
      allocate(SQI(3,Nq_irr));SQI(:,:)=0.0D0 
      ierr=CHDIR("./dir-eps") 
      do iq=1,Nq_irr 
       write(dirname,"('q',i3.3)")iq 
       ierr=CHDIR(dirname) 
       inquire(file='dat.log.400',exist=file_e) 
       if(file_e)then 
        write(6,*)'dat.log.400 exists in ',trim(dirname) 
       else
        write(6,*)'error: no dat.log.400 in ',trim(dirname) 
       endif 
       OPEN(301,FILE='dat.sq') 
       rewind(301)
       inquire(file='dat.sq',exist=file_e) 
       if(file_e)then 
        read(301,*)(SQI(i,iq),i=1,3) 
       else
        write(6,*)'error: no dat.sq in',trim(dirname) 
       endif 
       close(301) 
       ierr=CHDIR("..") 
      enddo!iq 
      ierr=CHDIR("..") 
      call system('pwd') 
!--
!gen(SQ,numirrq,numrotq,trsq,RWq) 
      NTQ=NTK!TOTAL NUMBER OF Q POINTS 
      allocate(SQ(3,NTQ));SQ(:,:)=0.0d0 
      allocate(numirrq(NTQ));numirrq(:)=0 
      allocate(numrotq(NTQ));numrotq(:)=0 
      allocate(trsq(NTQ));trsq(:)=0
      allocate(RWq(3,NTQ));RWq(:,:)=0      
      !--
      do iq=1,Nq_irr 
       SQ(:,iq)=SQI(:,iq) 
       numirrq(iq)=iq;numrotq(iq)=1;trsq(iq)=1;RWq(1:3,iq)=0
      enddo 
      jk=Nq_irr 
      do iq=1,Nq_irr
      do iop=1,Nsymq
      !--
      !sym
      !--
       ktmp(:)=0.0d0;RWtmp(:)=0  
       ktmp(1)=rg(1,1,iop)*SQI(1,iq)+rg(1,2,iop)*SQI(2,iq)
     +        +rg(1,3,iop)*SQI(3,iq)
       ktmp(2)=rg(2,1,iop)*SQI(1,iq)+rg(2,2,iop)*SQI(2,iq)
     +        +rg(2,3,iop)*SQI(3,iq)
       ktmp(3)=rg(3,1,iop)*SQI(1,iq)+rg(3,2,iop)*SQI(2,iq)
     +        +rg(3,3,iop)*SQI(3,iq)
       call kcheck(ktmp(1),RWtmp(1))!rewind check 
       do iik=1,jk 
        if(abs(SQ(1,iik)-ktmp(1))<1.0d-4.and. 
     +     abs(SQ(2,iik)-ktmp(2))<1.0d-4.and. 
     +     abs(SQ(3,iik)-ktmp(3))<1.0d-4)goto 1100
       enddo!iik
       jk=jk+1
       SQ(:,jk)=ktmp(:)
       numirrq(jk)=iq;numrotq(jk)=iop;trsq(jk)=1;RWq(:,jk)=RWtmp(:)
!time-reversal
1100   ktmp(:)=0.0d0;RWtmp(:)=0  
       ktmp(1)=rg(1,1,iop)*SQI(1,iq)+rg(1,2,iop)*SQI(2,iq)
     +        +rg(1,3,iop)*SQI(3,iq)
       ktmp(2)=rg(2,1,iop)*SQI(1,iq)+rg(2,2,iop)*SQI(2,iq)
     +        +rg(2,3,iop)*SQI(3,iq) 
       ktmp(3)=rg(3,1,iop)*SQI(1,iq)+rg(3,2,iop)*SQI(2,iq)
     +        +rg(3,3,iop)*SQI(3,iq) 
       call kcheck_trs(ktmp(1),RWtmp(1))!rewind check 20170322
       do iik=1,jk
        if(abs(SQ(1,iik)-(-ktmp(1)))<1.0d-4.and. 
     +     abs(SQ(2,iik)-(-ktmp(2)))<1.0d-4.and. 
     +     abs(SQ(3,iik)-(-ktmp(3)))<1.0d-4)goto 2100
       enddo!iik
       jk=jk+1
       SQ(:,jk)=-ktmp(:)
       numirrq(jk)=iq;numrotq(jk)=iop;trsq(jk)=-1;RWq(:,jk)=RWtmp(:)
2100  enddo 
      enddo 
      !--
      if(NTQ/=jk)then 
       write(6,*)'ERROR;STOP;NTQ should be jk'   
       write(6,*)'NTQ=',NTQ,'jk=',jk;STOP
      endif 
      !--
      write(6,*) 
      write(6,*)'============'
      write(6,*)' SQ and SK0 '
      write(6,*)'============'
      write(6,*) 
      do iq=1,NTQ 
       write(6,'(I5,3F10.5,2x,3F10.5)') iq,SQ(:,iq),SK0(:,iq) 
      enddo 
!--
!gen(NGQ_eps,NGQ_psi,LG0)
      allocate(LG0(3,NTG,NTQ));LG0(:,:,:)=0
      allocate(NGQ_eps(NTQ));NGQ_eps(:)=0
      allocate(NGQ_psi(NTQ));NGQ_psi(:)=0
      !--
      do iq=1,Nq_irr 
       q1=SQI(1,iq);q2=SQI(2,iq);q3=SQI(3,iq)
       call make_LG0(NTG,b1(1),b2(1),b3(1),Ecut_for_eps,
     +      Ecut_for_psi,q1,q2,q3,LG0(1,1,iq),NG_for_eps,NG_for_psi) 
       NGQ_eps(iq)=NG_for_eps;NGQ_psi(iq)=NG_for_psi  
       write(6,'(i8,3f10.5,a8,i8,a8,i10)') 
     + iq,q1,q2,q3,'NGeps',NG_for_eps,'NGpsi',NG_for_psi  
      enddo 
      do iq=Nq_irr+1,NTQ 
       if(trsq(iq)==1)then 
        ik=numirrq(iq);iop=numrotq(iq) 
        q1=rg(1,1,iop)*SQI(1,ik)+rg(1,2,iop)*SQI(2,ik)
     +    +rg(1,3,iop)*SQI(3,ik)+dble(RWq(1,iq)) 
        q2=rg(2,1,iop)*SQI(1,ik)+rg(2,2,iop)*SQI(2,ik)
     +    +rg(2,3,iop)*SQI(3,ik)+dble(RWq(2,iq))  
        q3=rg(3,1,iop)*SQI(1,ik)+rg(3,2,iop)*SQI(2,ik)
     +    +rg(3,3,iop)*SQI(3,ik)+dble(RWq(3,iq))  
        call make_LG0(NTG,b1(1),b2(1),b3(1),Ecut_for_eps,
     +       Ecut_for_psi,q1,q2,q3,LG0(1,1,iq),NG_for_eps,NG_for_psi) 
        NGQ_eps(iq)=NG_for_eps;NGQ_psi(iq)=NG_for_psi  
        write(6,'(i8,3f10.5,a8,i8,a8,i10)') 
     +  iq,q1,q2,q3,'NGeps',NG_for_eps,'NGpsi',NG_for_psi  
       elseif(trsq(iq)==-1)then  
        ik=numirrq(iq); iop=numrotq(iq) 
        q1=rg(1,1,iop)*SQI(1,ik)+rg(1,2,iop)*SQI(2,ik)
     +    +rg(1,3,iop)*SQI(3,ik)+dble(RWq(1,iq)) 
        q2=rg(2,1,iop)*SQI(1,ik)+rg(2,2,iop)*SQI(2,ik)
     +    +rg(2,3,iop)*SQI(3,ik)+dble(RWq(2,iq))  
        q3=rg(3,1,iop)*SQI(1,ik)+rg(3,2,iop)*SQI(2,ik)
     +    +rg(3,3,iop)*SQI(3,ik)+dble(RWq(3,iq))  
        KGtmp(:,:)=0 
        call make_LG0(NTG,b1(1),b2(1),b3(1),Ecut_for_eps,
     +       Ecut_for_psi,q1,q2,q3,KGtmp(1,1),NG_for_eps,NG_for_psi) 
        NGQ_eps(iq)=NG_for_eps;NGQ_psi(iq)=NG_for_psi  
        write(6,'(i8,3f10.5,a8,i8,a8,i10)') 
     +  iq,q1,q2,q3,'NGeps',NG_for_eps,'NGpsi',NG_for_psi  
        LG0(:,:,iq)=-KGtmp(:,:) 
       endif 
      enddo 
      !--
      Lq1=maxval(abs(LG0(1,:,:)))+1;write(6,*)'Lq1=',Lq1 
      Lq2=maxval(abs(LG0(2,:,:)))+1;write(6,*)'Lq2=',Lq2 
      Lq3=maxval(abs(LG0(3,:,:)))+1;write(6,*)'Lq3=',Lq3 
      allocate(packingq(-Lq1:Lq1,-Lq2:Lq2,-Lq3:Lq3,Nq_irr))   
      packingq(:,:,:,:)=0 
      do iq=1,Nq_irr 
       do ig=1,NGQ_eps(iq) 
        i1=LG0(1,ig,iq);j1=LG0(2,ig,iq);k1=LG0(3,ig,iq) 
        packingq(i1,j1,k1,iq)=ig 
       enddo 
      enddo 
      !-- 
      NTGQ=maxval(NGQ_eps(:))!TOTAL NUMBER OF G VECTORS FOR EPSILON
      write(6,*)'NTGQ=',NTGQ  
!--
!OPEN(600-,R,FILE='DATA_EPSILON_fRPA',FORM='unformatted') 
      allocate(epstmp(NTGQ,NTGQ,ne));epstmp(:,:,:)=0.0D0!real8
      allocate(epstmpgm(NTGQ,NTGQ,ne,3));epstmpgm(:,:,:,:)=0.0D0!real8
      allocate(epsirr(NTGQ,NTGQ,ne,Nq_irr));epsirr(:,:,:,:)=0.0D0!real4 
      !--
      ierr=CHDIR("./dir-eps") 
      call system('pwd') 
      do iq=1,Nq_irr 
       if(abs(SKI(1,iq))<1.0d-5.and.abs(SKI(2,iq))<1.0d-5.and.
     +    abs(SKI(3,iq))<1.0d-5)then 
        write(dirname,"('q',i3.3)")iq
        iqgm=iq 
        ierr=CHDIR(dirname) 
        do ix=1,3 
         file_num=600+(ix-1)  
         write(filename,'("dat.epsqw.",i3.3)')file_num 
         open(file_num,file=filename,form='unformatted') 
         write(6,'(a10,a15,a5,a10)') 
     +   'read: ',trim(filename),'in ',trim(dirname)  
         rewind(file_num) 
         NG_for_eps=NGQ_eps(iq)
         read(file_num)(((epstmpgm(ig,jg,ie,ix),ig=1,NG_for_eps)
     +                          ,jg=1,NG_for_eps),ie=1,ne)
        enddo!ix 
        epsirr(:,:,:,iq)!epsirr:real4,epstmp:real8 
     + =(epstmpgm(:,:,:,1)+epstmpgm(:,:,:,2)+epstmpgm(:,:,:,3))/3.0d0
       else
        write(dirname,"('q',i3.3)")iq
        ierr=CHDIR(dirname) 
        file_num=600 
        write(filename,'("dat.epsqw.",i3.3)')file_num 
        open(file_num,file=filename,form='unformatted') 
        write(6,'(a10,a15,a5,a10)') 
     +  'read: ',trim(filename),'in ',trim(dirname)  
        rewind(file_num) 
        NG_for_eps=NGQ_eps(iq)
        read(file_num)(((epstmp(ig,jg,ie),ig=1,NG_for_eps)
     +                         ,jg=1,NG_for_eps),ie=1,ne)
        epsirr(:,:,:,iq)=epstmp(:,:,:)!epsirr:real4,epstmp:real8 
       endif!q=0 or not 
       ierr=CHDIR("..") 
      enddo!iq   
      ierr=CHDIR("..") 
      call system('pwd') 
!--
!      do iq=1,Nq_irr 
!       NG_for_eps=NGQ_eps(iq)
!       write(6,*) NG_for_eps 
!       do ig=1,NG_for_eps 
!        write(6,*) epsirr(ig,ig,100,iq) 
!       enddo  
!      enddo  
!--
!make W4(iw,jw,kw,lw,R)
      WRITE(6,*) 
      WRITE(6,*)'====================================='
      WRITE(6,*)'=== INTERACTION CALCULATION START ==='
      WRITE(6,*)'====================================='
      WRITE(6,*) 
      !--
      !CONSIDER SINGLE FREQUENCY           
      !--
      ie=calc_ifreq 
      write(6,*)'wgrid_No=',ie  
      !--
      allocate(i_ij(NWF*NWF));i_ij=0 
      allocate(j_ij(NWF*NWF));j_ij=0   
      ip=0 
      do iw=1,NWF
       do jw=1,NWF 
        ip=ip+1
        i_ij(ip)=iw
        j_ij(ip)=jw  
       enddo 
      enddo 
      write(6,*)'total-pair=',ip  
      !--
      allocate(W4(NWF,NWF,NWF,NWF,-Na1:Na1,-Na2:Na2,-Na3:Na3));W4=0.0d0 
      allocate(func(NWF,NWF,NWF,NWF,NTQ));func=0.0d0  
      allocate(X4(NWF,NWF,NWF,NWF,-Na1:Na1,-Na2:Na2,-Na3:Na3));X4=0.0d0 
      allocate(fbr(NWF,NWF,NWF,NWF,NTQ));fbr=0.0d0  
      !--
      !CONSIDER R=0
      !--
      do ia1=0,0!-Na1,Na1 
       do ia2=0,0!-Na2,Na2
        do ia3=0,0!-Na3,Na3
         do iq=1,NTQ 
          q1=SQ(1,iq);q2=SQ(2,iq);q3=SQ(3,iq)
          !--
          !q/=0 
          !--
          if(q1.ne.0.0d0.or.q2.ne.0.0d0.or.q3.ne.0.0d0)then 
           NG_for_eps=NGQ_eps(iq);NG_for_psi=NGQ_psi(iq)
           allocate(length_qg(NG_for_psi));length_qg(:)=0.0D0 
           do igL=1,NG_for_psi   
            igL1=LG0(1,igL,iq)
            igL2=LG0(2,igL,iq)
            igL3=LG0(3,igL,iq)
            qgL(:)=(q1+dble(igL1))*b1(:)
     +            +(q2+dble(igL2))*b2(:)
     +            +(q3+dble(igL3))*b3(:)
            qgL2=qgL(1)**2+qgL(2)**2+qgL(3)**2
            qgL1=dsqrt(qgL2) 
            length_qg(igL)=qgL1
           enddo!igL 
           allocate(rho(NWF,NWF,NG_for_psi));rho(:,:,:)=0.0D0 
!$OMP PARALLEL PRIVATE(ik,ip,iw,jw,pf,shift_G,ikq,C0_K,C0_KQ,
!$OMP& rho_tmp,wfunc,fftwk,prho)
           allocate(prho(NWF,NWF,NG_for_psi));prho(:,:,:)=0.0D0 
           allocate(C0_K(NTG));C0_K(:)=0.0d0     
           allocate(C0_KQ(NTG));C0_KQ(:)=0.0d0     
           allocate(fftwk(Nl123*2));fftwk(:)=0.0d0  
           allocate(wfunc(Nl123*2));wfunc(:)=0.0d0 
           allocate(rho_tmp(NG_for_psi))
!$OMP DO 
           do ik=1,NTK 
            do ip=1,NWF*NWF 
             iw=i_ij(ip)
             jw=j_ij(ip)  
             pf=exp(-ci*tpi*(SK0(1,ik)*dble(ia1)
     +                      +SK0(2,ik)*dble(ia2)
     +                      +SK0(3,ik)*dble(ia3))) 
             shift_G(:)=0
             call search_kq(NTK,SK0,q1,q2,q3,ik,ikq,shift_G)
             C0_K(:)=C0(:,jw,ik) 
             C0_KQ(:)=C0(:,iw,ikq) 
             rho_tmp(:)=0.0D0 
             call calc_InterStateMatrix(NTK,NTG,NG0(1),KG0(1,1,1),
     +       C0_K(1),C0_KQ(1),ik,ikq,nwx2,nwy2,nwz2,nfft1,nfft2, 
     +       Nl123,wfunc(1),fftwk(1),fs,LG0(1,1,iq),NG_for_psi,
     +       shift_G(1),rho_tmp(1))
             prho(iw,jw,:)=prho(iw,jw,:)+rho_tmp(:)*pf/length_qg(:)   
            enddo!ip
           enddo!ik 
!$OMP END DO 
!$OMP CRITICAL 
           rho=rho+prho 
!$OMP END CRITICAL 
           deallocate(prho,fftwk,wfunc,rho_tmp,C0_K,C0_KQ) 
!$OMP END PARALLEL 
           rho(:,:,:)=rho(:,:,:)/dble(NTK)
           deallocate(length_qg)
           !-- 
           allocate(epsmk(NTGQ,NTGQ,ne));epsmk(:,:,:)=0.0d0 
           allocate(packtmp(-Lq1:Lq1,-Lq2:Lq2,-Lq3:Lq3)) 
           iqir=numirrq(iq)
           iop=numrotq(iq)
           packtmp(:,:,:)=packingq(:,:,:,iqir) 
           call make_eps(NTG,NTGQ,ne,trsq(iq),NGQ_eps(iq),LG0(1,1,iq), 
     +     RWq(1,iq),rginv(1,1,iop),pg(1,iop),Lq1,Lq2,Lq3,
     +     packtmp(-Lq1,-Lq2,-Lq3),epsirr(1,1,1,iqir),epsmk(1,1,1)) 
!--
!$OMP PARALLEL PRIVATE(ip,iw,jw,jp,kw,lw,SUM_CMPX,igL,jgL)
!$OMP DO 
           do ip=1,NWF*NWF  
            iw=i_ij(ip)
            jw=j_ij(ip) 
            do jp=1,NWF*NWF  
             kw=i_ij(jp)
             lw=j_ij(jp) 
             SUM_CMPX=0.0D0 
             do igL=1,NG_for_eps 
              do jgL=1,NG_for_eps 
               SUM_CMPX
     +        =SUM_CMPX
     +        +rho(iw,jw,igL)*epsmk(igL,jgL,ie)*CONJG(rho(lw,kw,jgL)) 
              enddo 
             enddo 
             do igL=NG_for_eps+1,NG_for_psi  
              SUM_CMPX=SUM_CMPX+rho(iw,jw,igL)*CONJG(rho(lw,kw,igL))   
             enddo 
             func(iw,jw,kw,lw,iq)=SUM_CMPX 
             !--
             !bare
             !--
             SUM_CMPX=0.0D0 
             do igL=1,NG_for_psi  
              SUM_CMPX=SUM_CMPX+rho(iw,jw,igL)*CONJG(rho(lw,kw,igL))   
             enddo 
             fbr(iw,jw,kw,lw,iq)=SUM_CMPX 
            enddo!jp  
           enddo!ip  
!$OMP END DO 
!$OMP END PARALLEL 
           deallocate(epsmk,packtmp)  
           deallocate(rho)  
           write(6,*)'finish iq=',iq 
          !--
          !q=0 
          !--
          elseif(q1==0.0d0.and.q2==0.0d0.and.q3==0.0d0) then 
           write(6,'(a,x,3f10.5)')'q1,q2,q3',q1,q2,q3 
           NG_for_eps=NGQ_eps(iq); NG_for_psi=NGQ_psi(iq)
           allocate(length_qg(NG_for_psi));length_qg(:)=0.0D0 
           do igL=1,NG_for_psi 
            igL1=LG0(1,igL,iq)
            igL2=LG0(2,igL,iq)
            igL3=LG0(3,igL,iq)
            if(igL1==0.and.igL2==0.and.igL3==0) then 
             NoG0=igL  
             write(6,*)'NoG0=',NoG0
            else
             qgL(:)=(q1+dble(igL1))*b1(:)+(q2+dble(igL2))*b2(:)
     +             +(q3+dble(igL3))*b3(:)
             qgL2=qgL(1)**2+qgL(2)**2+qgL(3)**2
             qgL1=dsqrt(qgL2) 
             length_qg(igL)=qgL1
            endif  
           enddo!igL 
           allocate(rho(NWF,NWF,NG_for_psi));rho(:,:,:)=0.0D0 
!$OMP PARALLEL PRIVATE(ik,ip,iw,jw,pf,shift_G,ikq,C0_K,C0_KQ,
!$OMP& rho_tmp,wfunc,fftwk,prho)
           allocate(prho(NWF,NWF,NG_for_psi));prho(:,:,:)=0.0D0 
           allocate(C0_K(NTG));C0_K(:)=0.0d0     
           allocate(C0_KQ(NTG));C0_KQ(:)=0.0d0     
           allocate(fftwk(Nl123*2));fftwk(:)=0.0d0  
           allocate(wfunc(Nl123*2));wfunc(:)=0.0d0 
           allocate(rho_tmp(NG_for_psi))
!$OMP DO 
           do ik=1,NTK 
            do ip=1,NWF*NWF 
             iw=i_ij(ip)
             jw=j_ij(ip)  
             pf=exp(-ci*tpi*(SK0(1,ik)*dble(ia1)
     +                      +SK0(2,ik)*dble(ia2)
     +                      +SK0(3,ik)*dble(ia3))) 
             ikq=ik 
             shift_G(:)=0 
             rho_tmp(:)=0.0D0 
             C0_K(:)=C0(:,jw,ik) 
             C0_KQ(:)=C0(:,iw,ikq) 
             rho_tmp(:)=0.0D0 
             call calc_InterStateMatrix(NTK,NTG,NG0(1),KG0(1,1,1),
     +       C0_K(1),C0_KQ(1),ik,ikq,nwx2,nwy2,nwz2,nfft1,nfft2, 
     +       Nl123,wfunc(1),fftwk(1),fs,LG0(1,1,iq),NG_for_psi,
     +       shift_G(1),rho_tmp(1))
!---20151122--------------------!   
             rho_tmp(NoG0)=0.0D0 
!-------------------------------!
             prho(iw,jw,:)=prho(iw,jw,:)+rho_tmp(:)*pf  
            enddo!ip  
           enddo!ik 
!$OMP END DO 
!$OMP CRITICAL 
           rho=rho+prho 
!$OMP END CRITICAL 
           deallocate(prho,rho_tmp,fftwk,wfunc,C0_K,C0_KQ) 
!$OMP END PARALLEL 
           do ip=1,NWF*NWF 
            iw=i_ij(ip);jw=j_ij(ip)  
            do igL=1,NG_for_psi  
             if(igL/=NoG0) then 
              rho(iw,jw,igL)=rho(iw,jw,igL)/length_qg(igL)/dble(NTK)
             else 
              rho(iw,jw,igL)=rho(iw,jw,igL)/dble(NTK)  
             endif 
            enddo!igL 
            write(6,*)'rho(NoG0)=',rho(iw,jw,NoG0) 
           enddo!ip  
           deallocate(length_qg)
           !--
           allocate(epsmk(NTGQ,NTGQ,ne));epsmk(:,:,:)=0.0d0 
           allocate(packtmp(-Lq1:Lq1,-Lq2:Lq2,-Lq3:Lq3));packtmp=0.0d0  
           iqir=numirrq(iq)
           iop=numrotq(iq)
           packtmp(:,:,:)=packingq(:,:,:,iqir) 
           call make_eps(NTG,NTGQ,ne,trsq(iq),NGQ_eps(iq),LG0(1,1,iq), 
     +     RWq(1,iq),rginv(1,1,iop),pg(1,iop),Lq1,Lq2,Lq3,
     +     packtmp(-Lq1,-Lq2,-Lq3),epsirr(1,1,1,iqir),epsmk(1,1,1)) 
!head-eff(Louie)
           qsz=(6.0D0*(pi**2)/dble(NTQ)/dble(VOLUME))**(1.0D0/3.0D0)  
           chead=(2.0D0/pi)*epsmk(NoG0,NoG0,ie)*qsz 
           write(6,*)'correction_head(cRPA) in eV=',chead*au    
!head-bare(Louie)
           chead_bare=(2.0D0/pi)*qsz 
           write(6,*)'correction_head(bare) in eV=',chead_bare*au    
!--
!wing  
!        SUM_CMPX=0.0D0 
!        do igL=1,NG_for_eps 
!         SUM_CMPX=SUM_CMPX+rho(iw,jw,igL)*epsmk(igL,NoG0,ie)
!        enddo 
!        cwingL=(dsqrt(2.0d0)*tpi*Rc/dble(NTQ)/VOLUME)*SUM_CMPX 
!        write(6,*)'cwingL in eV=',cwingL*au   
!        SUM_CMPX=0.0D0 
!        do igL=1,NG_for_eps 
!         SUM_CMPX=SUM_CMPX+CONJG(rho(lw,kw,igL))*epsmk(NoG0,igL,ie)
!        enddo 
!        cwingR=(dsqrt(2.0d0)*tpi*Rc/dble(NTQ)/VOLUME)*SUM_CMPX 
!        write(6,*)'cwingR in eV=',cwingR*au 
!--
!$OMP PARALLEL PRIVATE(ip,iw,jw,jp,kw,lw,SUM_CMPX,igL,jgL)
!$OMP DO 
           do ip=1,NWF*NWF  
            iw=i_ij(ip)
            jw=j_ij(ip) 
            do jp=1,NWF*NWF  
             kw=i_ij(jp)
             lw=j_ij(jp) 
             SUM_CMPX=0.0D0 
             do igL=1,NG_for_eps 
              do jgL=1,NG_for_eps 
               SUM_CMPX
     +        =SUM_CMPX
     +        +rho(iw,jw,igL)*epsmk(igL,jgL,ie)*CONJG(rho(lw,kw,jgL)) 
              enddo 
             enddo 
             do igL=NG_for_eps+1,NG_for_psi  
              SUM_CMPX=SUM_CMPX+rho(iw,jw,igL)*CONJG(rho(lw,kw,igL)) 
             enddo 
             func(iw,jw,kw,lw,iq)=SUM_CMPX 
             !--
             !bare
             !--
             SUM_CMPX=0.0D0 
             do igL=1,NG_for_psi  
              SUM_CMPX=SUM_CMPX+rho(iw,jw,igL)*CONJG(rho(lw,kw,igL)) 
             enddo 
             fbr(iw,jw,kw,lw,iq)=SUM_CMPX 
            enddo!jp  
           enddo!ip  
!$OMP END DO 
!$OMP END PARALLEL 
           deallocate(epsmk,packtmp)  
           deallocate(rho)  
           write(6,*)'finish iq=',iq 
          endif!q=0.or.q/=0
         enddo!iq 
         !--
         !eff
         !--
         do ip=1,NWF*NWF  
          iw=i_ij(ip);jw=j_ij(ip) 
          do jp=1,NWF*NWF  
           kw=i_ij(jp);lw=j_ij(jp) 
           SUM_CMPX=0.0D0  
           do iq=1,NTQ 
            SUM_CMPX=SUM_CMPX+func(iw,jw,kw,lw,iq) 
           enddo 
           W4(iw,jw,kw,lw,ia1,ia2,ia3)
     +    =2.0d0*tpi*SUM_CMPX/VOLUME/DBLE(NTQ) 
          enddo!jp  
         enddo!ip  
         !--
         !bare
         !--
         do ip=1,NWF*NWF  
          iw=i_ij(ip);jw=j_ij(ip) 
          do jp=1,NWF*NWF  
           kw=i_ij(jp);lw=j_ij(jp) 
           SUM_CMPX=0.0D0  
           do iq=1,NTQ 
            SUM_CMPX=SUM_CMPX+fbr(iw,jw,kw,lw,iq) 
           enddo 
           X4(iw,jw,kw,lw,ia1,ia2,ia3)
     +    =2.0d0*tpi*SUM_CMPX/VOLUME/DBLE(NTQ) 
          enddo!jp  
         enddo!ip  
        enddo!ia3      
       enddo!ia2 
      enddo!ia1      
      !--
      !head 
      !--
      do iw=1,NWF 
       do kw=1,NWF 
        W4(iw,iw,kw,kw,0,0,0)=W4(iw,iw,kw,kw,0,0,0)+chead       
        X4(iw,iw,kw,kw,0,0,0)=X4(iw,iw,kw,kw,0,0,0)+chead_bare       
       enddo 
      enddo 
!--
!write X4
!--
      write(6,*) 
      write(6,'(a)')'- Bare 4-center at R=0 and w=calc_ifreq' 
      write(6,*) 
      do ia1=0,0!-Na1,Na1 
      do ia2=0,0!-Na2,Na2
      do ia3=0,0!-Na3,Na3
       !write(6,'(a6,3i5)')'R:',ia1,ia2,ia3           
       write(6,'(4a5,a14)')'i','j','k','l','W_ijkl [eV]'
       write(6,'(a34)')'----------------------------------'
       do ip=1,NWF*NWF  
        iw=i_ij(ip);jw=j_ij(ip) 
        do jp=1,NWF*NWF  
         kw=i_ij(jp);lw=j_ij(jp) 
!         write(6,'(a20,4i5,f15.10)')'<ij|X|kl>',iw,jw,kw,lw,
!     +   dble(W4(iw,jw,kw,lw,ia1,ia2,ia3))*au  
         write(6,'(4i5,f15.10)')iw,jw,kw,lw,
     +   dble(X4(iw,jw,kw,lw,ia1,ia2,ia3))*au  
        enddo 
       enddo  
       write(6,*)  
      enddo  
      enddo  
      enddo  
!--
!write W4 
!--
      write(6,*) 
      write(6,'(a)')'- Screened 4-center at R=0 & w=calc_ifreq' 
      write(6,*) 
      do ia1=0,0!-Na1,Na1 
      do ia2=0,0!-Na2,Na2
      do ia3=0,0!-Na3,Na3
       !write(6,'(a6,3i5)')'R:',ia1,ia2,ia3           
       write(6,'(4a5,a14)')'i','j','k','l','W_ijkl [eV]'
       write(6,'(a34)')'----------------------------------'
       do ip=1,NWF*NWF  
        iw=i_ij(ip);jw=j_ij(ip) 
        do jp=1,NWF*NWF  
         kw=i_ij(jp);lw=j_ij(jp) 
!         write(6,'(a20,4i5,f15.10)')'<ij|W|kl>',iw,jw,kw,lw,
!     +   dble(W4(iw,jw,kw,lw,ia1,ia2,ia3))*au  
         write(6,'(4i5,f15.10)')iw,jw,kw,lw,
     +   dble(W4(iw,jw,kw,lw,ia1,ia2,ia3))*au  
        enddo 
       enddo  
       write(6,*)  
      enddo  
      enddo  
      enddo  
      !--
      STOP
      END           
