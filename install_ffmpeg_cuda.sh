#
# build configuration uses same options as default ffmpeg for Ubuntu 20.04 + cuda
#
sudo apt install -y \
  cmake \
  flite1-dev \
  frei0r-plugins-dev \
  ladspa-sdk \
  libaom-dev \
  libass-dev \
  libavc1394-dev \
  libbluray-dev \
  libbs2b-dev \
  libcaca-dev \
  libcdio-dev \
  libcdio-paranoia-dev \
  libchromaprint-dev \
  libcodec2-dev \
  libdc1394-dev \
  libdrm-dev \
  libffmpeg-nvenc-dev \
  libgme-dev \
  libgnutls28-dev \
  libgsm1-dev \
  libiec61883-dev \
  libjack-dev \
  liblilv-dev \
  libmp3lame-dev \
  libmysofa-dev \
  libnvidia-encode-470 \
  libomxil-bellagio-dev \
  libopenal-dev \
  libopenjp2-7-dev \
  libopenmpt-dev \
  libopus-dev \
  libpulse-dev \
  librsvg2-dev \
  librubberband-dev \
  libsdl2-dev \
  libshine-dev \
  libsnappy-dev \
  libsoxr-dev \
  libspeex-dev \
  libssh-dev \
  libtheora-dev \
  libtwolame-dev \
  libunistring-dev \
  libvidstab-dev \
  libvpx-dev \
  libwebp-dev \
  libx264-dev \
  libx265-dev \
  libxvidcore-dev \
  libzmq3-dev \
  libzvbi-dev \
  nvidia-cuda-toolkit \
  yasm 

git clone --branch master --depth 1 git@github.com:FFmpeg/nv-codec-headers.git && \
  cd nv-codec-headers && \
  make && \
  sudo make install && \
  cd .. && \
  rm -Rf nv-codec-headers

git clone --branch master --depth 1 git@github.com:AviSynth/AviSynthPlus.git && \
  cd AviSynthPlus && \
  mkdir avisynth-build && \
  cd avisynth-build && \
  cmake ../ -DHEADERS_ONLY:bool=on && \
  sudo make VersionGen install && 
  cd ../.. &&
  rm -Rf AviSynthPlus
  
git clone --branch master --depth 1 git@github.com:FFmpeg/FFmpeg.git && \
  cd FFmpeg 

./configure \
  --cpu=host \
  --extra-cflags="-march=native" \
  --disable-stripping \
  --enable-avisynth \
  --enable-chromaprint \
  --enable-cuda-nvcc \
  --enable-cuvid \
  --enable-frei0r \
  --enable-gnutls \
  --enable-gpl \
  --enable-ladspa \
  --enable-libaom \
  --enable-libass \
  --enable-libbluray \
  --enable-libbs2b \
  --enable-libcaca \
  --enable-libcdio \
  --enable-libcodec2 \
  --enable-libdc1394 \
  --enable-libdrm \
  --enable-libflite \
  --enable-libfontconfig \
  --enable-libfreetype \
  --enable-libfribidi \
  --enable-libgme \
  --enable-libgsm \
  --enable-libiec61883 \
  --enable-libjack \
  --enable-libmp3lame \
  --enable-libmysofa \
  --enable-libnpp \
  --enable-libopenjpeg \
  --enable-libopenmpt \
  --enable-libopus \
  --enable-libpulse \
  --enable-librsvg \
  --enable-librubberband \
  --enable-libshine \
  --enable-libsnappy \
  --enable-libsoxr \
  --enable-libspeex \
  --enable-libssh \
  --enable-libtheora \
  --enable-libtwolame \
  --enable-libvidstab \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libwebp \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libxml2 \
  --enable-libxvid \
  --enable-libzmq\
  --enable-libzvbi \
  --enable-lv2 \
  --enable-nonfree \
  --enable-nvenc \
  --enable-omx \
  --enable-openal \
  --enable-opencl \
  --enable-opengl \
  --enable-sdl2 \
  --enable-shared \
  --extra-version=1ubuntu0.1 \
  --incdir=/usr/include/x86_64-linux-gnu \
  --libdir=/usr/lib/x86_64-linux-gnu \
  --prefix=/usr \
  --toolchain=hardened
  
# install prefix            /usr
# source path               .
# C compiler                gcc
# C library                 glibc
# ARCH                      x86 (znver2)
# version string suffix     1ubuntu0.1
# big-endian                no
# runtime cpu detection     yes
# standalone assembly       yes
# x86 assembler             yasm
# MMX enabled               yes
# MMXEXT enabled            yes
# 3DNow! enabled            yes
# 3DNow! extended enabled   yes
# SSE enabled               yes
# SSSE3 enabled             yes
# AESNI enabled             yes
# AVX enabled               yes
# AVX2 enabled              yes
# AVX-512 enabled           yes
# AVX-512ICL enabled        yes
# XOP enabled               yes
# FMA3 enabled              yes
# FMA4 enabled              yes
# i686 features enabled     yes
# CMOV is fast              yes
# EBX available             yes
# EBP available             yes
# debug symbols             yes
# strip symbols             no
# optimize for size         no
# optimizations             yes
# static                    yes
# shared                    yes
# postprocessing support    yes
# network support           yes
# threading support         pthreads
# safe bitstream reader     yes
# texi2html enabled         no
# perl enabled              yes
# pod2man enabled           yes
# makeinfo enabled          no
# makeinfo supports HTML    no
# xmllint enabled           no
# 
# External libraries:
# alsa                    iconv                   libcaca                 libfontconfig           libjack                 libpulse                libspeex                libvpx                  libxml2                 openal
# avisynth                ladspa                  libcdio                 libfreetype             libmp3lame              librsvg                 libssh                  libwebp                 libxvid                 opengl
# bzlib                   libaom                  libcodec2               libfribidi              libmysofa               librubberband           libtheora               libx264                 libzmq                  sdl2
# chromaprint             libass                  libdc1394               libgme                  libopenjpeg             libshine                libtwolame              libx265                 libzvbi                 sndio
# frei0r                  libbluray               libdrm                  libgsm                  libopenmpt              libsnappy               libvidstab              libxcb                  lv2                     xlib
# gnutls                  libbs2b                 libflite                libiec61883             libopus                 libsoxr                 libvorbis               libxcb_shm              lzma                    zlib
# 
# External libraries providing hardware acceleration:
# cuda                    cuda_nvcc               ffnvcodec               nvdec                   omx                     v4l2_m2m
# cuda_llvm               cuvid                   libnpp                  nvenc                   opencl                  vdpau
# 
# Libraries:
# avcodec                 avdevice                avfilter                avformat                avutil                  postproc                swresample              swscale
# 
# Programs:
# ffmpeg                  ffplay                  ffprobe
# 
# Enabled decoders:
# aac                     adpcm_psx               bfi                     dvdsub                  h263p                   libvorbis               msa1                    pcm_s24le_planar        rawvideo                targa                   vp8_v4l2m2m
# aac_fixed               adpcm_sbpro_2           bink                    dvvideo                 h264                    libvpx_vp8              mscc                    pcm_s32be               realtext                targa_y216              vp9
# aac_latm                adpcm_sbpro_3           binkaudio_dct           dxa                     h264_cuvid              libvpx_vp9              msmpeg4v1               pcm_s32le               rl2                     tdsc                    vp9_cuvid
# aasc                    adpcm_sbpro_4           binkaudio_rdft          dxtory                  h264_v4l2m2m            libzvbi_teletext        msmpeg4v2               pcm_s32le_planar        roq                     text                    vp9_v4l2m2m
# ac3                     adpcm_swf               bintext                 dxv                     hap                     loco                    msmpeg4v3               pcm_s64be               roq_dpcm                theora                  vplayer
# ac3_fixed               adpcm_thp               bitpacked               eac3                    hca                     lscr                    msnsiren                pcm_s64le               rpza                    thp                     vqa
# acelp_kelvin            adpcm_thp_le            bmp                     eacmv                   hcom                    m101                    msp2                    pcm_s8                  rscc                    tiertexseqvideo         vqc
# adpcm_4xm               adpcm_vima              bmv_audio               eamad                   hdr                     mace3                   msrle                   pcm_s8_planar           rv10                    tiff                    wavpack
# adpcm_adx               adpcm_xa                bmv_video               eatgq                   hevc                    mace6                   mss1                    pcm_sga                 rv20                    tmv                     wbmp
# adpcm_afc               adpcm_yamaha            bonk                    eatgv                   hevc_cuvid              magicyuv                mss2                    pcm_u16be               rv30                    truehd                  wcmv
# adpcm_agm               adpcm_zork              brender_pix             eatqi                   hevc_v4l2m2m            mdec                    msvideo1                pcm_u16le               rv40                    truemotion1             webp
# adpcm_aica              agm                     c93                     eightbps                hnm4_video              media100                mszh                    pcm_u24be               s302m                   truemotion2             webvtt
# adpcm_argo              aic                     cavs                    eightsvx_exp            hq_hqa                  metasound               mts2                    pcm_u24le               sami                    truemotion2rt           wmalossless
# adpcm_ct                alac                    ccaption                eightsvx_fib            hqx                     microdvd                mv30                    pcm_u32be               sanm                    truespeech              wmapro
# adpcm_dtk               alias_pix               cdgraphics              escape124               huffyuv                 mimic                   mvc1                    pcm_u32le               sbc                     tscc                    wmav1
# adpcm_ea                als                     cdtoons                 escape130               hymt                    misc4                   mvc2                    pcm_u8                  scpr                    tscc2                   wmav2
# adpcm_ea_maxis_xa       amrnb                   cdxl                    evrc                    iac                     mjpeg                   mvdv                    pcm_vidc                screenpresso            tta                     wmavoice
# adpcm_ea_r1             amrwb                   cfhd                    exr                     idcin                   mjpeg_cuvid             mvha                    pcx                     sdx2_dpcm               twinvq                  wmv1
# adpcm_ea_r2             amv                     cinepak                 fastaudio               idf                     mjpegb                  mwsc                    pfm                     sga                     txd                     wmv2
# adpcm_ea_r3             anm                     clearvideo              ffv1                    iff_ilbm                mlp                     mxpeg                   pgm                     sgi                     ulti                    wmv3
# adpcm_ea_xas            ansi                    cljr                    ffvhuff                 ilbc                    mmvideo                 nellymoser              pgmyuv                  sgirle                  utvideo                 wmv3image
# adpcm_g722              apac                    cllc                    ffwavesynth             imc                     mobiclip                notchlc                 pgssub                  sheervideo              v210                    wnv1
# adpcm_g726              ape                     comfortnoise            fic                     imm4                    motionpixels            nuv                     pgx                     shorten                 v210x                   wrapped_avframe
# adpcm_g726le            apng                    cook                    fits                    imm5                    movtext                 on2avc                  phm                     simbiosis_imx           v308                    ws_snd1
# adpcm_ima_acorn         aptx                    cpia                    flac                    indeo2                  mp1                     opus                    photocd                 sipr                    v408                    xan_dpcm
# adpcm_ima_alp           aptx_hd                 cri                     flashsv                 indeo3                  mp1float                paf_audio               pictor                  siren                   v410                    xan_wc3
# adpcm_ima_amv           arbc                    cscd                    flashsv2                indeo4                  mp2                     paf_video               pixlet                  smackaud                vb                      xan_wc4
# adpcm_ima_apc           argo                    cyuv                    flic                    indeo5                  mp2float                pam                     pjs                     smacker                 vble                    xbin
# adpcm_ima_apm           ass                     dca                     flv                     interplay_acm           mp3                     pbm                     png                     smc                     vbn                     xbm
# adpcm_ima_cunning       asv1                    dds                     fmvc                    interplay_dpcm          mp3adu                  pcm_alaw                ppm                     smvjpeg                 vc1                     xface
# adpcm_ima_dat4          asv2                    derf_dpcm               fourxm                  interplay_video         mp3adufloat             pcm_bluray              prores                  snow                    vc1_cuvid               xl
# adpcm_ima_dk3           atrac1                  dfa                     fraps                   ipu                     mp3float                pcm_dvd                 prosumer                sol_dpcm                vc1_v4l2m2m             xma1
# adpcm_ima_dk4           atrac3                  dfpwm                   frwu                    jacosub                 mp3on4                  pcm_f16le               psd                     sonic                   vc1image                xma2
# adpcm_ima_ea_eacs       atrac3al                dirac                   ftr                     jpeg2000                mp3on4float             pcm_f24le               ptx                     sp5x                    vcr1                    xpm
# adpcm_ima_ea_sead       atrac3p                 dnxhd                   g2m                     jpegls                  mpc7                    pcm_f32be               qcelp                   speedhq                 vmdaudio                xsub
# adpcm_ima_iss           atrac3pal               dolby_e                 g723_1                  jv                      mpc8                    pcm_f32le               qdm2                    speex                   vmdvideo                xwd
# adpcm_ima_moflex        atrac9                  dpx                     g729                    kgv1                    mpeg1_cuvid             pcm_f64be               qdmc                    srgc                    vmnc                    y41p
# adpcm_ima_mtf           aura                    dsd_lsbf                gdv                     kmvc                    mpeg1_v4l2m2m           pcm_f64le               qdraw                   srt                     vorbis                  ylc
# adpcm_ima_oki           aura2                   dsd_lsbf_planar         gem                     lagarith                mpeg1video              pcm_lxf                 qoi                     ssa                     vp3                     yop
# adpcm_ima_qt            av1                     dsd_msbf                gif                     libaom_av1              mpeg2_cuvid             pcm_mulaw               qpeg                    stl                     vp4                     yuv4
# adpcm_ima_rad           av1_cuvid               dsd_msbf_planar         gremlin_dpcm            libcodec2               mpeg2_v4l2m2m           pcm_s16be               qtrle                   subrip                  vp5                     zero12v
# adpcm_ima_smjpeg        avrn                    dsicinaudio             gsm                     libgsm                  mpeg2video              pcm_s16be_planar        r10k                    subviewer               vp6                     zerocodec
# adpcm_ima_ssi           avrp                    dsicinvideo             gsm_ms                  libgsm_ms               mpeg4                   pcm_s16le               r210                    subviewer1              vp6a                    zlib
# adpcm_ima_wav           avs                     dss_sp                  h261                    libopenjpeg             mpeg4_cuvid             pcm_s16le_planar        ra_144                  sunrast                 vp6f                    zmbv
# adpcm_ima_ws            avui                    dst                     h263                    libopus                 mpeg4_v4l2m2m           pcm_s24be               ra_288                  svq1                    vp7
# adpcm_ms                ayuv                    dvaudio                 h263_v4l2m2m            librsvg                 mpegvideo               pcm_s24daud             ralf                    svq3                    vp8
# adpcm_mtaf              bethsoftvid             dvbsub                  h263i                   libspeex                mpl2                    pcm_s24le               rasc                    tak                     vp8_cuvid
# 
# Enabled encoders:
# a64multi                adpcm_yamaha            dca                     h263                    libopus                 mp2                     pcm_f64be               pcm_u16le               r10k                    subrip                  wbmp
# a64multi5               alac                    dfpwm                   h263_v4l2m2m            libshine                mp2fixed                pcm_f64le               pcm_u24be               r210                    sunrast                 webvtt
# aac                     alias_pix               dnxhd                   h263p                   libspeex                mpeg1video              pcm_mulaw               pcm_u24le               ra_144                  svq1                    wmav1
# ac3                     amv                     dpx                     h264_nvenc              libtheora               mpeg2video              pcm_s16be               pcm_u32be               rawvideo                targa                   wmav2
# ac3_fixed               apng                    dvbsub                  h264_omx                libtwolame              mpeg4                   pcm_s16be_planar        pcm_u32le               roq                     text                    wmv1
# adpcm_adx               aptx                    dvdsub                  h264_v4l2m2m            libvorbis               mpeg4_omx               pcm_s16le               pcm_u8                  roq_dpcm                tiff                    wmv2
# adpcm_argo              aptx_hd                 dvvideo                 hap                     libvpx_vp8              mpeg4_v4l2m2m           pcm_s16le_planar        pcm_vidc                rpza                    truehd                  wrapped_avframe
# adpcm_g722              ass                     eac3                    hdr                     libvpx_vp9              msmpeg4v2               pcm_s24be               pcx                     rv10                    tta                     xbm
# adpcm_g726              asv1                    exr                     hevc_nvenc              libwebp                 msmpeg4v3               pcm_s24daud             pfm                     rv20                    ttml                    xface
# adpcm_g726le            asv2                    ffv1                    hevc_v4l2m2m            libwebp_anim            msvideo1                pcm_s24le               pgm                     s302m                   utvideo                 xsub
# adpcm_ima_alp           avrp                    ffvhuff                 huffyuv                 libx264                 nellymoser              pcm_s24le_planar        pgmyuv                  sbc                     v210                    xwd
# adpcm_ima_amv           avui                    fits                    jpeg2000                libx264rgb              opus                    pcm_s32be               phm                     sgi                     v308                    y41p
# adpcm_ima_apm           ayuv                    flac                    jpegls                  libx265                 pam                     pcm_s32le               png                     smc                     v408                    yuv4
# adpcm_ima_qt            bitpacked               flashsv                 libaom_av1              libxvid                 pbm                     pcm_s32le_planar        ppm                     snow                    v410                    zlib
# adpcm_ima_ssi           bmp                     flashsv2                libcodec2               ljpeg                   pcm_alaw                pcm_s64be               prores                  sonic                   vbn                     zmbv
# adpcm_ima_wav           cfhd                    flv                     libgsm                  magicyuv                pcm_bluray              pcm_s64le               prores_aw               sonic_ls                vc2
# adpcm_ima_ws            cinepak                 g723_1                  libgsm_ms               mjpeg                   pcm_dvd                 pcm_s8                  prores_ks               speedhq                 vorbis
# adpcm_ms                cljr                    gif                     libmp3lame              mlp                     pcm_f32be               pcm_s8_planar           qoi                     srt                     vp8_v4l2m2m
# adpcm_swf               comfortnoise            h261                    libopenjpeg             movtext                 pcm_f32le               pcm_u16be               qtrle                   ssa                     wavpack
# 
# Enabled hwaccels:
# av1_nvdec               h264_vdpau              hevc_vdpau              mpeg1_nvdec             mpeg2_nvdec             mpeg4_nvdec             vc1_nvdec               vp8_nvdec               vp9_vdpau               wmv3_vdpau
# h264_nvdec              hevc_nvdec              mjpeg_nvdec             mpeg1_vdpau             mpeg2_vdpau             mpeg4_vdpau             vc1_vdpau               vp9_nvdec               wmv3_nvdec
# 
# Enabled parsers:
# aac                     avs2                    dca                     dvbsub                  g729                    hdr                     mlp                     pnm                     tak                     webp
# aac_latm                avs3                    dirac                   dvd_nav                 gif                     hevc                    mpeg4video              qoi                     vc1                     xbm
# ac3                     bmp                     dnxhd                   dvdsub                  gsm                     ipu                     mpegaudio               rv30                    vorbis                  xma
# adx                     cavsvideo               dolby_e                 flac                    h261                    jpeg2000                mpegvideo               rv40                    vp3                     xwd
# amr                     cook                    dpx                     ftr                     h263                    misc4                   opus                    sbc                     vp8
# av1                     cri                     dvaudio                 g723_1                  h264                    mjpeg                   png                     sipr                    vp9
# 
# Enabled demuxers:
# aa                      asf_o                   dash                    fwse                    image_dpx_pipe          image_xwd_pipe          mlp                     nuv                     r3d                     smush                   vobsub
# aac                     ass                     data                    g722                    image_exr_pipe          imf                     mlv                     obu                     rawvideo                sol                     voc
# aax                     ast                     daud                    g723_1                  image_gem_pipe          ingenient               mm                      ogg                     realtext                sox                     vpk
# ac3                     au                      dcstr                   g726                    image_gif_pipe          ipmovie                 mmf                     oma                     redspark                spdif                   vplayer
# ace                     av1                     derf                    g726le                  image_hdr_pipe          ipu                     mods                    paf                     rl2                     srt                     vqf
# acm                     avi                     dfa                     g729                    image_j2k_pipe          ircam                   moflex                  pcm_alaw                rm                      stl                     w64
# act                     avisynth                dfpwm                   gdv                     image_jpeg_pipe         iss                     mov                     pcm_f32be               roq                     str                     wav
# adf                     avr                     dhav                    genh                    image_jpegls_pipe       iv8                     mp3                     pcm_f32le               rpl                     subviewer               wc3
# adp                     avs                     dirac                   gif                     image_jpegxl_pipe       ivf                     mpc                     pcm_f64be               rsd                     subviewer1              webm_dash_manifest
# ads                     avs2                    dnxhd                   gsm                     image_pam_pipe          ivr                     mpc8                    pcm_f64le               rso                     sup                     webvtt
# adx                     avs3                    dsf                     gxf                     image_pbm_pipe          jacosub                 mpegps                  pcm_mulaw               rtp                     svag                    wsaud
# aea                     bethsoftvid             dsicin                  h261                    image_pcx_pipe          jv                      mpegts                  pcm_s16be               rtsp                    svs                     wsd
# afc                     bfi                     dss                     h263                    image_pfm_pipe          kux                     mpegtsraw               pcm_s16le               s337m                   swf                     wsvqa
# aiff                    bfstm                   dts                     h264                    image_pgm_pipe          kvag                    mpegvideo               pcm_s24be               sami                    tak                     wtv
# aix                     bink                    dtshd                   hca                     image_pgmyuv_pipe       laf                     mpjpeg                  pcm_s24le               sap                     tedcaptions             wv
# alp                     binka                   dv                      hcom                    image_pgx_pipe          libgme                  mpl2                    pcm_s32be               sbc                     thp                     wve
# amr                     bintext                 dvbsub                  hevc                    image_phm_pipe          libopenmpt              mpsub                   pcm_s32le               sbg                     threedostr              xa
# amrnb                   bit                     dvbtxt                  hls                     image_photocd_pipe      live_flv                msf                     pcm_s8                  scc                     tiertexseq              xbin
# amrwb                   bitpacked               dxa                     hnm                     image_pictor_pipe       lmlm4                   msnwc_tcp               pcm_u16be               scd                     tmv                     xmv
# anm                     bmv                     ea                      ico                     image_png_pipe          loas                    msp                     pcm_u16le               sdp                     truehd                  xvag
# apac                    boa                     ea_cdata                idcin                   image_ppm_pipe          lrc                     mtaf                    pcm_u24be               sdr2                    tta                     xwma
# apc                     bonk                    eac3                    idf                     image_psd_pipe          luodat                  mtv                     pcm_u24le               sds                     tty                     yop
# ape                     brstm                   epaf                    iff                     image_qdraw_pipe        lvf                     musx                    pcm_u32be               sdx                     txd                     yuv4mpegpipe
# apm                     c93                     ffmetadata              ifv                     image_qoi_pipe          lxf                     mv                      pcm_u32le               segafilm                ty
# apng                    caf                     filmstrip               ilbc                    image_sgi_pipe          m4v                     mvi                     pcm_u8                  ser                     v210
# aptx                    cavsvideo               fits                    image2                  image_sunrast_pipe      matroska                mxf                     pcm_vidc                sga                     v210x
# aptx_hd                 cdg                     flac                    image2_alias_pix        image_svg_pipe          mca                     mxg                     pjs                     shorten                 vag
# aqtitle                 cdxl                    flic                    image2_brender_pix      image_tiff_pipe         mcc                     nc                      pmp                     siff                    vc1
# argo_asf                cine                    flv                     image2pipe              image_vbn_pipe          mgsts                   nistsphere              pp_bnk                  simbiosis_imx           vc1t
# argo_brp                codec2                  fourxm                  image_bmp_pipe          image_webp_pipe         microdvd                nsp                     pva                     sln                     vividas
# argo_cvg                codec2raw               frm                     image_cri_pipe          image_xbm_pipe          mjpeg                   nsv                     pvf                     smacker                 vivo
# asf                     concat                  fsb                     image_dds_pipe          image_xpm_pipe          mjpeg_2000              nut                     qcp                     smjpeg                  vmd
# 
# Enabled muxers:
# a64                     ass                     data                    framecrc                hls                     md5                     mpeg2vob                pcm_f32le               pcm_u32le               smjpeg                  uncodedframecrc
# ac3                     ast                     daud                    framehash               ico                     microdvd                mpegts                  pcm_f64be               pcm_u8                  smoothstreaming         vc1
# adts                    au                      dfpwm                   framemd5                ilbc                    mjpeg                   mpjpeg                  pcm_f64le               pcm_vidc                sox                     vc1t
# adx                     avi                     dirac                   g722                    image2                  mkvtimestamp_v2         mxf                     pcm_mulaw               psp                     spdif                   voc
# aiff                    avif                    dnxhd                   g723_1                  image2pipe              mlp                     mxf_d10                 pcm_s16be               rawvideo                spx                     w64
# alp                     avm2                    dts                     g726                    ipod                    mmf                     mxf_opatom              pcm_s16le               rm                      srt                     wav
# amr                     avs2                    dv                      g726le                  ircam                   mov                     null                    pcm_s24be               roq                     stream_segment          webm
# amv                     avs3                    eac3                    gif                     ismv                    mp2                     nut                     pcm_s24le               rso                     streamhash              webm_chunk
# apm                     bit                     f4v                     gsm                     ivf                     mp3                     obu                     pcm_s32be               rtp                     sup                     webm_dash_manifest
# apng                    caf                     ffmetadata              gxf                     jacosub                 mp4                     oga                     pcm_s32le               rtp_mpegts              swf                     webp
# aptx                    cavsvideo               fifo                    h261                    kvag                    mpeg1system             ogg                     pcm_s8                  rtsp                    tee                     webvtt
# aptx_hd                 chromaprint             fifo_test               h263                    latm                    mpeg1vcd                ogv                     pcm_u16be               sap                     tg2                     wsaud
# argo_asf                codec2                  filmstrip               h264                    lrc                     mpeg1video              oma                     pcm_u16le               sbc                     tgp                     wtv
# argo_cvg                codec2raw               fits                    hash                    m4v                     mpeg2dvd                opus                    pcm_u24be               scc                     truehd                  wv
# asf                     crc                     flac                    hds                     matroska                mpeg2svcd               pcm_alaw                pcm_u24le               segafilm                tta                     yuv4mpegpipe
# asf_stream              dash                    flv                     hevc                    matroska_audio          mpeg2video              pcm_f32be               pcm_u32be               segment                 ttml
# 
# Enabled protocols:
# async                   concatf                 file                    hls                     icecast                 libzmq                  pipe                    rtmpt                   subfile                 udp
# bluray                  crypto                  ftp                     http                    ipfs                    md5                     prompeg                 rtmpts                  tcp                     udplite
# cache                   data                    gopher                  httpproxy               ipns                    mmsh                    rtmp                    rtp                     tee                     unix
# concat                  ffrtmphttp              gophers                 https                   libssh                  mmst                    rtmps                   srtp                    tls
# 
# Enabled filters:
# a3dscope                alphaextract            asuperpass              colorchart              detelecine              framepack               kerndeint               nnedi                   replaygain              sidedata                tonemap
# abench                  alphamerge              asuperstop              colorcontrast           dialoguenhance          framerate               kirsch                  noformat                reverse                 sierpinski              tonemap_opencl
# abitscope               amerge                  atadenoise              colorcorrect            dilation                framestep               ladspa                  noise                   rgbashift               signalstats             tpad
# acompressor             ametadata               atempo                  colorhold               dilation_opencl         freezedetect            lagfun                  normalize               rgbtestsrc              signature               transpose
# acontrast               amix                    atilt                   colorize                displace                freezeframes            latency                 null                    roberts                 silencedetect           transpose_npp
# acopy                   amovie                  atrim                   colorkey                dnn_classify            frei0r                  lenscorrection          nullsink                roberts_opencl          silenceremove           transpose_opencl
# acrossfade              amplify                 avectorscope            colorkey_opencl         dnn_detect              frei0r_src              life                    nullsrc                 rotate                  sinc                    treble
# acrossover              amultiply               avgblur                 colorlevels             dnn_processing          fspp                    limitdiff               openclsrc               rubberband              sine                    tremolo
# acrusher                anequalizer             avgblur_opencl          colormap                doubleweave             gblur                   limiter                 oscilloscope            sab                     siti                    trim
# acue                    anlmdn                  avsynctest              colormatrix             drawbox                 geq                     loop                    overlay                 scale                   smartblur               unpremultiply
# addroi                  anlmf                   axcorrelate             colorspace              drawgraph               gradfun                 loudnorm                overlay_cuda            scale2ref               smptebars               unsharp
# adeclick                anlms                   azmq                    colorspace_cuda         drawgrid                gradients               lowpass                 overlay_opencl          scale2ref_npp           smptehdbars             unsharp_opencl
# adeclip                 anoisesrc               bandpass                colorspectrum           drawtext                graphmonitor            lowshelf                owdenoise               scale_cuda              sobel                   untile
# adecorrelate            anull                   bandreject              colortemperature        drmeter                 grayworld               lumakey                 pad                     scale_npp               sobel_opencl            v360
# adelay                  anullsink               bass                    compand                 dynaudnorm              greyedge                lut                     pad_opencl              scdet                   sofalizer               vaguedenoiser
# adenorm                 anullsrc                bbox                    compensationdelay       earwax                  guided                  lut1d                   pal100bars              scharr                  spectrumsynth           varblur
# aderivative             apad                    bench                   concat                  ebur128                 haas                    lut2                    pal75bars               scroll                  speechnorm              vectorscope
# adrawgraph              aperms                  bilateral               convolution             edgedetect              haldclut                lut3d                   palettegen              segment                 split                   vflip
# adynamicequalizer       aphasemeter             bilateral_cuda          convolution_opencl      elbg                    haldclutsrc             lutrgb                  paletteuse              select                  spp                     vfrdet
# adynamicsmooth          aphaser                 biquad                  convolve                entropy                 hdcd                    lutyuv                  pan                     selectivecolor          sr                      vibrance
# aecho                   aphaseshift             bitplanenoise           copy                    epx                     headphone               lv2                     perms                   sendcmd                 ssim                    vibrato
# aemphasis               apsyclip                blackdetect             cover_rect              eq                      hflip                   mandelbrot              perspective             separatefields          stereo3d                vidstabdetect
# aeval                   apulsator               blackframe              crop                    equalizer               highpass                maskedclamp             phase                   setdar                  stereotools             vidstabtransform
# aevalsrc                arealtime               blend                   cropdetect              erosion                 highshelf               maskedmax               photosensitivity        setfield                stereowiden             vif
# aexciter                aresample               blockdetect             crossfeed               erosion_opencl          hilbert                 maskedmerge             pixdesctest             setparams               streamselect            vignette
# afade                   areverse                blurdetect              crystalizer             estdif                  histeq                  maskedmin               pixelize                setpts                  subtitles               virtualbass
# afftdn                  arnndn                  bm3d                    cue                     exposure                histogram               maskedthreshold         pixscope                setrange                super2xsai              vmafmotion
# afftfilt                asdr                    boxblur                 curves                  extractplanes           hqdn3d                  maskfun                 pp                      setsar                  superequalizer          volume
# afifo                   asegment                boxblur_opencl          datascope               extrastereo             hqx                     mcompand                pp7                     settb                   surround                volumedetect
# afir                    aselect                 bs2b                    dblur                   fade                    hstack                  median                  premultiply             sharpen_npp             swaprect                vstack
# afirsrc                 asendcmd                bwdif                   dcshift                 feedback                hsvhold                 mergeplanes             prewitt                 shear                   swapuv                  w3fdif
# aformat                 asetnsamples            cas                     dctdnoiz                fftdnoiz                hsvkey                  mestimate               prewitt_opencl          showcqt                 tblend                  waveform
# afreqshift              asetpts                 cellauto                deband                  fftfilt                 hue                     metadata                program_opencl          showfreqs               telecine                weave
# afwtdn                  asetrate                channelmap              deblock                 field                   huesaturation           midequalizer            pseudocolor             showinfo                testsrc                 xbr
# agate                   asettb                  channelsplit            decimate                fieldhint               hwdownload              minterpolate            psnr                    showpalette             testsrc2                xcorrelate
# agraphmonitor           ashowinfo               chorus                  deconvolve              fieldmatch              hwmap                   mix                     pullup                  showspatial             thistogram              xfade
# ahistogram              asidedata               chromahold              dedot                   fieldorder              hwupload                monochrome              qp                      showspectrum            threshold               xfade_opencl
# aiir                    asoftclip               chromakey               deesser                 fifo                    hwupload_cuda           morpho                  random                  showspectrumpic         thumbnail               xmedian
# aintegral               aspectralstats          chromakey_cuda          deflate                 fillborders             hysteresis              movie                   readeia608              showvolume              thumbnail_cuda          xstack
# ainterleave             asplit                  chromanr                deflicker               find_rect               identity                mpdecimate              readvitc                showwaves               tile                    yadif
# alatency                ass                     chromashift             dejudder                firequalizer            idet                    mptestsrc               realtime                showwavespic            tiltshelf               yadif_cuda
# alimiter                astats                  ciescope                delogo                  flanger                 il                      msad                    remap                   shuffleframes           tinterlace              yaepblur
# allpass                 astreamselect           codecview               derain                  flite                   inflate                 multiply                remap_opencl            shufflepixels           tlut2                   yuvtestsrc
# allrgb                  asubboost               color                   deshake                 floodfill               interlace               negate                  removegrain             shuffleplanes           tmedian                 zmq
# allyuv                  asubcut                 colorbalance            deshake_opencl          format                  interleave              nlmeans                 removelogo              sidechaincompress       tmidequalizer           zoompan
# aloop                   asupercut               colorchannelmixer       despill                 fps                     join                    nlmeans_opencl          repeatfields            sidechaingate           tmix
# 
# Enabled bsfs:
# aac_adtstoasc           chomp                   dv_error_marker         h264_metadata           hevc_metadata           mjpega_dump_header      mpeg4_unpack_bframes    pcm_rechunk             setts                   vp9_metadata
# av1_frame_merge         dca_core                eac3_core               h264_mp4toannexb        hevc_mp4toannexb        mov2textsub             noise                   pgs_frame_merge         text2movsub             vp9_raw_reorder
# av1_frame_split         dts2pts                 extract_extradata       h264_redundant_pps      imx_dump_header         mp3_header_decompress   null                    prores_metadata         trace_headers           vp9_superframe
# av1_metadata            dump_extradata          filter_units            hapqa_extract           mjpeg2jpeg              mpeg2_metadata          opus_metadata           remove_extradata        truehd_core             vp9_superframe_split
# 
# Enabled indevs:
# alsa                    iec61883                kmsgrab                 libcdio                 openal                  pulse                   v4l2
# fbdev                   jack                    lavfi                   libdc1394               oss                     sndio                   xcbgrab
# 
# Enabled outdevs:
# alsa                    caca                    fbdev                   opengl                  oss                     pulse                   sdl2                    sndio                   v4l2                    xv
# 
# License: nonfree and unredistributable
  
make -j 16 && \
  sudo make install
  
ffmpeg -version

# ffmpeg version git-2022-10-18-3141dbb-1ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers
# built with gcc 9 (Ubuntu 9.4.0-1ubuntu1~20.04.1)
# configuration: --cpu=host --extra-cflags='-march=native' --disable-stripping --enable-avisynth --enable-chromaprint --enable-cuda-nvcc --enable-cuvid --enable-frei0r --enable-gnutls --enable-gpl --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdc1394 --enable-libdrm --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libiec61883 --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libnpp --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-nonfree --enable-nvenc --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-shared --extra-version=1ubuntu0.1 --incdir=/usr/include/x86_64-linux-gnu --libdir=/usr/lib/x86_64-linux-gnu --prefix=/usr --toolchain=hardened
# libavutil      57. 39.101 / 57. 39.101
# libavcodec     59. 51.100 / 59. 51.100
# libavformat    59. 34.101 / 59. 34.101
# libavdevice    59.  8.101 / 59.  8.101
# libavfilter     8. 49.101 /  8. 49.101
# libswscale      6.  8.112 /  6.  8.112
# libswresample   4.  9.100 /  4.  9.100
# libpostproc    56.  7.100 / 56.  7.100


  