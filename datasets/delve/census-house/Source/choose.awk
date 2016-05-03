BEGIN{
FS="\t"
  ind=0;
#  while(getline line < "fields"){
#    ind++;
#     names[line]=ind;
#  }
#  close ("fields");i=1;
#  str="P001000" i;
#  print names[str];
}

function divField(fieldStr,from,To,divisor)
{

  if(divisor==0) {
    printf("line%d: str=%s: div=0; exit \n",FNR,fieldStr) > "/dev/stderr";
    exit;}
  for(ii=from;ii<=To;ii++) {
    jj=getPos(fieldStr,ii)
    if(jj==0) {
      printf("divField: jj=0; fieldStr=%s\n",fieldStr) > "/dev/stderr"; exit;}
    $jj=$jj/divisor;
  }
}
# ------------------------------------------------------------ #
function collapseField(fieldStr, lims, npieces) {
  for(i=1;i<=npieces;i++){
    from=lims[2*(i-1)+1];
    To=lims[2*i];
    Res=0;
    for(ii=from;ii<=To; ii++) {
      jj=getPos(fieldStr,ii)
      if(jj==0) {
	printf("collapseField1: jj=0; str=%s, ii=%d\n",str,ii) > "/dev/stderr"; 
	exit;
      }
      whichToPrint[jj]=0;
      Res+=$jj;
    }
    whereTo=getPos(fieldStr,i)
    if(whereTo==0){
      printf("collapseField: whereTo=0; str=%s\n",str) > "/dev/stderr"; 
	exit;
    }
    $whereTo=Res;
    whichToPrint[whereTo]=1;
      
  }
}


# ------------------------------------------------------------ #

function deleteFields(fieldStr,lims,npieces)
{
  
  for(i=1;i<=npieces;i++){
    from=lims[2*(i-1)+1];
    To=lims[2*i];
    ii=from;
    for(ii=from;ii<=To; ii++) {
      jj=getPos(fieldStr,ii)
      if(jj==0) {
	printf("deleteFields: jj=0; fieldStr=%s\n",fieldStr) > "/dev/stderr"; 
	exit;
      }
      whichToPrint[jj]=0;
    }
  }
}

# ------------------------------------------------------------ #

function getPos(str,subfield)
{
  if(subfield<=9){
    fullStr=str "000" subfield;
    return names[fullStr]
  }
  else {
    fullStr=str "00" subfield;
    return names[fullStr]
  }

}
    
# ------------------------------------------------------------ #    

function averageFields(fieldStr,weights,from,To,where)
{

  Res=0;weInd=0;
  for(ii=from;ii<=To; ii++) {
    weInd++;
    jj=getPos(fieldStr,ii)
    if(jj==0) {
      printf("averageFields: jj=0; str=%s\n",str) > "/dev/stderr"; 
	exit;
      }
    Res=Res+weights[weInd]*($jj);
    whichToPrint[jj]=0;
  }
    whereTo=getPos(fieldStr,where)
  if(whereTo==0) {
    printf("averageFields: whereTo=0; where=%d\n",where) > "/dev/stderr"; 
    exit;
  }
  $whereTo=Res;
  whichToPrint[whereTo]=1;
}





############################################################
#
### End of utility functions
#
############################################################

{
  if (NF!=468 ) {
#    printf("line=%d - not all fields present (NF=%d)\n",FNR,NF) > "/dev/stderr";
    next;
  }
}

$0 ~ /^FIPS/ {
  for(ind=1;ind<=NF;ind++)
    names[$ind]=ind;
}

$0 !~ /^FIPS/  { 
  for(i=1;i<=NF;i++)
    whichToPrint[i]=1;

  which=names["P0010001"]; persons=$which; if(persons==0) next;
  which=names["P0020001"]; famil=$which;if(famil==0)next;
  which=names["P0030001"]; housh=$which;if(housh==0)next;
  which=names["P0050001"]; males=$which;if(males==0)males=1;
  which=names["P0050002"]; females=$which;if(females==0)females=1;
  which=names["P0060001"]; whites=$which;if(whites==0)whites=1;
  which=names["P0060002"]; blacks=$which;
  which=names["P0060003"]; indians =$which; 
  which=names["P0060004"]; asians =$which; 
  which=names["P0060005"]; other =$which; 
#  printf("\nline%d\t%d pers; %d famil; %d HH-olds; %d males; %d fem; %d whites; %d bla; %d indi; %d asia; %d other\n",FNR,persons,famil,housh,males,females, whites, blacks, indians,  asians, other);
  if(blacks==0) blacks=1;
  if(indians==0) indians=1;
  if(asians==0) asians=1;
  if(other==0) other=1;
  under18=0;
  for(i=1;i<=9;i++){
    str="P011000" i
      which=names[str];
    under18+=$which;
  }
  for(i=10;i<=12;i++){
    str="P01100" i
      which=names[str];
    under18+=$which;
  }
  if(under18==0) under18=1;
  
  
  
  
### P4Urban/Rural. Univ=Persons. 1 .. 4
  divField("P004",1,4,persons);
  
### P5 Sex
  divField("P005",1,2,persons);
  
### P6 Race 1 .. 5
  divField("P006",1,5,persons);
  
### P8 Hispanic 1 .. 1
  divField("P008",1,1,persons);
  
### P11 Age 1 .. 31
  divField("P011",1,31,persons);
  lims[1]=1;lims[2]=7;
  lims[3]=8;lims[4]=17;
  lims[5]=18;lims[6]=26;
  lims[7]=27;lims[8]=31;
  collapseField("P011",lims,4);
  
# P14 Sex by Marital status 2x5
  divField("P014",1,5,males);
  divField("P014",6,10,females);
  
# P15 HH type and Relation  1..13
  divField("P015",1,13,persons);
  lims[1]=1;lims[2]=7;
  lims[3]=8;lims[4]=10;
  lims[5]=11;lims[6]=13;
  collapseField("P015",lims,3);
  
  
# P16 HH size and Type 1..10
  divField("P016",1,10,housh);
  lims[1]=1;lims[2]=2;
  lims[3]=3;lims[4]=8;
  lims[5]=9;lims[6]=10;
  collapseField("P016",lims,3);
  
  
  
# P17 Persons in Families 1..1
  divField("P017",1,1,persons);
  
# P18 Age of HH Members by HH Type 1..10
  divField("P018",1,10,housh);
  lims[1]=1;lims[2]=3;
  lims[3]=4;lims[4]=5;
  lims[5]=6;lims[6]=8;
  lims[7]=9;lims[8]=10
    collapseField("P018",lims,4);
  
  
# P19 Race by HH type (as % of HH-lds)
  divField("P019",1,40,housh); 
  lims[1]=1;lims[2]=8;
  lims[3]=9;lims[4]=16;
  
  lims[5]=17;lims[6]=24;
  lims[7]=25;lims[8]=32;
  
  lims[9]=33;lims[10]=40;
  
  collapseField("P019",lims,5);
  
  
  
  
# P20 Household Types with Hispanic Householder (as % of total housholds)
  divField("P020",1,8,housh);   
  lims[1]=1;lims[2]=8;
  collapseField("P020",lims,1);
  
  
# P21 HouseHold Type and Relation for Persons under 18 (as % of under 18)
#divField("P021",1,9,under18); 
  lims[1]=1;lims[2]=9;
  deleteFields("P021",lims,1);
  
# P24 Age(< or > 60) of HH members(2) by HH size and type(3) (as a %
# of HH-olds) 
#divField("P024",1,6,housh); 
  lims[1]=1;lims[2]=6;
  deleteFields("P024",lims,1);
  
  
# P25 Age(< or > 65) of HH members(2) by HH size and type(3) (as a %
# of HH-olds) 
  divField("P025",1,6,housh); 
  lims[1]=1;lims[2]=3;
  lims[3]=4;lims[4]=6;
  collapseField("P025",lims,2);
  
# P26 HH-old Type 
  divField("P026",1,2,housh); 
  
  
# P27 HH-old Type and Size 1..13
  divField("P027",1,13,housh); 
  lims[1]=1;lims[2]=3;
  lims[3]=4;lims[4]=6;
  lims[5]=7;lims[6]=7
    lims[7]=8;lims[8]=13;
  
  collapseField("P027",lims,4);
  
  lims[1]=1;lims[2]=2;
  deleteFields("P033",lims,1);
  
# ----------------------------------------------------------------------
  
  
  
# H1 Housing Units 
  which=names["H0010001"]; houses=$which; if(houses==0) next;
#  printf("houses %d->%d->%d\n",which,$which,vacant);
  which=names["H0020001"]; occupied=$which;if(occupied==0)occupied=1;
  which=names["H0020002"]; vacant=$which;
  if(vacant==0)vacant=1;
#  printf("vacant %d->%d->%d\n",which,$which,vacant);
  which=names["H0030001"]; ownOcc=$which;if(ownOcc==0)ownOcc=1;
#  printf("ownOcc %d->%d->%d\n",which,$which,ownOcc);
  which=names["H0030002"]; rentOcc=$which;if(rentOcc==0)rentOcc=1;
#  printf("rentOcc %d->%d->%d\n",which,$which,vacant);
  which=names["H0270001"]; specOwnOcc=$which;
#  printf("specOwnOcc %d->%d->%d\n",which,$which,vacant);
  which=names["H0270002"]; specOwnOcc+=$which;
#  printf("specOwnOcc %d->%d->%d\n",which,$which,vacant);
  if(specOwnOcc==0)specOwnOcc=1;
  
  which=names["H0050001"]; vacantForRent=$which;
  if(vacantForRent==0)vacantForRent=1;
  which=names["H0050002"]; vacantForSale=$which;
  if(vacantForSale==0)vacantForSale=1;
  
  
  
#printf("line=%d\t%d houses; %d occu; %d vac; %d OwnOcc; %d rentOc\n",FNR,houses,occupied,vacant,ownOcc,rentOcc);
  
  
  
# H2 Occupancy Status
  divField("H002",1,2,houses);
  
# H3 Tenure 
  divField("H003",1,2,occupied); 
  
# H4 Urban/Rural
  divField("H004",1,4,houses); 
  
  
# H5 Vacancy Status
  divField("H005",1,6,vacant); 
  
# H6 Boarded-up status (as % of vacant)
#divField("H006",1,2,houses); 
  lims[1]=1;lims[2]=2;
  deleteFields("H006",lims,1);
    
# H7 Usual Home elsewhere (as % of vacant)
  divField("H007",1,2,vacant); 
  lims[1]=2;lims[2]=2;
  deleteFields("H007",lims,1);
    
    
# H8 Race of HH-older (as % of occupied)
  divField("H008",1,5,occupied); 
  
  
# H9 - tenure by race of HH-older (as % of owner/renter occupied)
  divField("H009",1,5,ownOcc); 
  divField("H009",6,10,rentOcc); 
  
# H10 Hispanic origin of HH-older by race of HH-older (as % of occupied)  
  lims[1]=1;lims[2]=5;
  lims[3]=6;lims[4]=10;
  collapseField("H010",lims,2);
  divField("H010",1,2,occupied); 
  
# H11 Tenure by race of HH-older (as % of owner/renter occupied)
#divField("H011",1,5,ownOcc); 
#divField("H011",6,10,rentOcc); 
  lims[1]=1;lims[2]=10;
  deleteFields("H011",lims,1);
  
  
# H12 Tenure by age (as % of o/r occ)
  divField("H012",1,7,ownOcc); 
  divField("H012",8,14,rentOcc);   
  weights[1]=20;weights[2]=30;weights[3]=40;weights[4]=50;
  weights[5]=60;weights[6]=70;weights[7]=85;
  averageFields("H012",weights,1,7,1);
  averageFields("H012",weights,8,14,2);
  
# H13 % of housing units (HU) with x rooms
  lims[1]=1;lims[2]=4;lims[3]=5;lims[4]=9;
  collapseField("H013",lims,2);
  divField("H013",1,2,houses);   
  
  
# H14 Should I change aggregate rooms to average room per HU-->Yes
  divField("H014",1,1,houses);
  
  
  
# H15 Aggre rooms by tenure as an average num of rooms per ren/own
# occupied
  divField("H015",1,1,ownOcc);
  divField("H015",2,2,rentOcc);
  
  
# H16 aggre rooms by vacancy status as % of vacant
# divField("H016",1,6,vacant); 
#   for(ii=1;ii<=6;ii++) {
#     str="H016000" ii;
#     jj=names[str];
#     if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
  
#     $jj=$jj/vacant;
#   }
  lims[1]=1;lims[2]=6;
  deleteFields("H016",lims,1);
  
  
# H17 Persons in Units as a % of occup with x persons in
  lims[1]=1;lims[2]=4;
  lims[3]=5;lims[4]=7;
  collapseField("H017",lims,2);
  divField("H017",1,2,occupied); 
  
# H18 Tenure by persons occupied (as a % of occ/rent HU)
  divField("H018",1,7,ownOcc);
  divField("H018",8,14,rentOcc);  
  lims[1]=1;lims[2]=4;
  lims[3]=5;lims[4]=7;
  lims[5]=8;lims[6]=11;
  lims[7]=12;lims[8]=14;
  collapseField("H018",lims,4);
  
# H19 Aggregate Persons
    divField("H019",1,1,occupied); 
  
# H20 Aggre persons by tenure -> ratio of persons in OwnOcc/RentOcc
  lims[1]=1;lims[2]=2;
  deleteFields("H020",lims,1);

  
# H21 Persons per room as % of occ HU
#divField("H021",1,5,occupied); 
  lims[1]=1;lims[2]=5;
  deleteFields("H021",lims,1);
  
  
# H22 Persons per room by tenure (as % of own/rent occ HU)
#divField("H022",1,5,ownOcc);
#divField("H022",6,10,rentOcc);  
  lims[1]=1;lims[2]=10;
  deleteFields("H022",lims,1);
  
  
# H23 value as a % of specOwnOcc HU
  divField("H023",1,19,specOwnOcc); 
  lims[1]=1;lims[2]=8;
  lims[3]=9;lims[4]=11;
  lims[5]=12;lims[6]=16;
  lims[7]=17;lims[8]=20;
  collapseField("H023",lims,4);
  
  
# H24 Aggre Value as an average
  divField("H024",1,1,specOwnOcc); 
  
# H26 Agrre by race of HH-older (as average value per each race) 
# Must be done before we change the counts H25
  
  for(ii=1;ii<=5;ii++){
    str="H026000" ii;
    jj=names[str];
    if(jj==0) {printf("jj=0; str=%s\n",str)>"/dev/stderr"; exit;}
    str="H025000" ii;
    jj1=names[str];
    if(jj1==0) {printf("jj1=0; str=%s\n",str)>"/dev/stderr"; exit;}
    temp=$jj1;
    if(temp<1) {temp=1;}
    $jj=($jj)/temp;
  }
  
  
# H25 Race of HH-older (as % of spec Owner Occ HU)
#divField("H025",1,5,specOwnOcc); 
  lims[1]=1;lims[2]=5;
  deleteFields("H025",lims,1);
  
  
  
# H28 Aggre value by Hispanic origin (as aver value per HH-oder)
  
  for(ii=1;ii<=2;ii++){
    str="H028000" ii;
    jj=names[str];
    if(jj==0) {printf("jj=0; str=%s\n",str)>"/dev/stderr"; exit;}
    str="H027000" ii;
    jj1=names[str];
    if(jj1==0) {printf("jj1=0; str=%s\n",str)>"/dev/stderr"; exit;}
    temp=$jj1;
    if(temp<1) {temp=1;}
    $jj=($jj)/temp;
  }
  
# H27 Hispanic Origin of HH-older (as % of specOwnOcc)
  divField("H027",1,2,specOwnOcc); 
  lims[1]=1;lims[2]=2;
  deleteFields("H027",lims,1);
  
  
  
# H29 Aggregate value by units in structure as average price per unit
# of given size. Must get Num of units from H43 (collapse H43.4-.8)
  lims[1]=1;lims[2]=6;
  deleteFields("H029",lims,1);
  
#   for(ii=1;ii<=3;ii++) {
#     str="H029000" ii;
#     jj=names[str];
#     if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
#     str="H043000" ii;
#     jj1=names[str];
#     if(jj1==0) {printf("jj1=0; str=%s\n",str); exit;}
#     if($jj1==0)jj1=1 ;else jj1=$jj1;
#     $jj=$jj/jj1
#   }
# #collapse H43 for 3 or more
#   tot=0;
#   for(ii=4;ii<=8;ii++){
#     str="H043000" ii;
#     jj=names[str];
#     if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
#     tot+=$jj;
#   }
#   if(tot==0)tot=1;
#   str="H0290004";
#   jj=names[str];
#   if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
#   $jj/=tot;
# # get the rest  
#     str="H0290005";
#     jj=names[str];
#     if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
#     str="H0430009";
#     jj1=names[str];
#     if(jj1==0) {printf("jj1=0; str=%s\n",str); exit;}
#     if($jj1==0)jj1=1 ;else jj1=$jj1;
#     $jj=$jj/jj1
#     str="H0290006";
#     jj=names[str];
#     if(jj==0) {printf("jj=0; str=%s\n",str); exit;}
#     str="H0430010";
#     jj1=names[str];
#     if(jj1==0) {printf("jj1=0; str=%s\n",str); exit;}
#     if($jj1==0)jj1=1 ;else jj1=$jj1;
#     $jj=$jj/jj1
  
  
  
  
  which=names["H0300001"]; specVacForRent=$which; 
  which=names["H0300002"]; specVacForSale=$which;
  which=names["H0300003"]; specVacOther=$which;
  if(specVacForRent==0)specVacForRent=1;
  if(specVacForSale==0)specVacForSale=1;
  if(specVacForOther==0)specVacForOther=1;
  
  
  which=names["H0370001"];specRentOccCash=$which;
  which=names["H0370002"];
  specRentOccCash+=$which;
  if(specRentOccCash==0)specRentOccCash=1;
  
  
# H30 Vacancy status (as % of all vacant)
#divField("H030",1,3,vacant); 
  lims[1]=1;lims[2]=3;
  deleteFields("H030",lims,1);
  
  
# H31 Aggregate price asked (as average)
  divField("H031",1,1,specVacForSale); 
  
# H32 Contract rent (as percentage of total specrentoccu)

  lims[1]=1;lims[2]=17;
  deleteFields("H032",lims,1);
  
  
# H33 Agrre contract rent as average      
  divField("H033",1,1,specRentOccCash);   
  
  
# H35 Aggre contract rent by race of HH-older
  
  for(ii=1;ii<=5;ii++){
    str="H035000" ii;
    jj=names[str];
    if(jj==0) {printf("jj=0; str=%s\n",str)>"/dev/stderr"; exit;}
    str="H034000" ii;
    jj1=names[str];
    if(jj1==0) {printf("jj1=0; str=%s\n",str)>"/dev/stderr"; exit;}
    temp=$jj1;
    if(temp<1.0) {temp=1.0; }
    $jj=($jj)/temp;
  }
  
  
# H34 Race of HHolde as a % of specRentOccCash
#divField("H034",1,5,specRentOccCash); 
  lims[1]=1;lims[2]=5;
  deleteFields("H034",lims,1);
  
  
# H37 Contract rent by Hisp (as aver per HH-older of H/NH origin)
  for(ii=1;ii<=2;ii++){
    str="H037000" ii;
    jj=names[str];
    if(jj==0) {printf("jj=0; str=%s\n",str)>"/dev/stderr" ; exit;}
    str="H036000" ii;
    jj1=names[str];
    if(jj1==0) {printf("jj1=0; str=%s\n",str)>"/dev/stderr" ; exit;}
    temp=$jj1;
    if(temp<1) {temp=1;}
    $jj=($jj)/temp;
  }

  lims[1]=1;lims[2]=1;
  deleteFields("H037",lims,1);
  
# H36 Hispanic Origin of HHolde as a % of specRentOccCash
  divField("H036",1,2,specRentOccCash); 
  lims[1]=1;lims[2]=2;
  deleteFields("H036",lims,1);
  
# H38 Aggre rent asked (as average per VacantFor Rent)
  divField("H038",1,1,specVacForRent); 
  
# H39 Age of HH-lder (< or > 65) by meals included and cash rent
#   divField("H039",1,2,specRentOccCash); 
#   tot=specRentOcc-specRentOccCash;
#   if(tot==0)tot=1;
#   divField("H039",3,3,tot); 
#   divField("H039",4,5,specRentOccCash); 
#   divField("H039",6,6,tot); 
  
  lims[1]=1;lims[2]=6;
  deleteFields("H039",lims,1);
  
  
  
# H40 Vac-cy status by dur-ion of vac-cy (as % of all vacant)
  divField("H040",1,3,vacantForRent); 
  divField("H040",4,6,vacantForSale); 
  lims[1]=2;lims[2]=2;
  lims[3]=5;lims[4]=5;
  lims[5]=7;lims[6]=9;
  deleteFields("H040",lims,3);
  
  
# H41 Units in structure (as % of housing units in structures of size x)
#divField("H041",1,10,houses); 
  lims[1]=1;lims[2]=10;
  deleteFields("H041",lims,1);
  
  
# H42 Units in vacant structures (as % of vacant units in structures
# of size x)
#divField("H042",1,10,vacant); 
  lims[1]=1;lims[2]=10;
  deleteFields("H042",lims,1);
  
# H43 Tenure by units in structure (as percent of ownOcc)
#divField("H043",1,10,ownOcc); 
#divField("H043",11,20,rentOcc); 
  
  lims[1]=1;lims[2]=20;
  deleteFields("H043",lims,1);
  
  
# H44 Agrregate persons by tenure by units in structure (as ave person
# per HU) 
#divField("H044",1,10,ownOcc); 
#divField("H044",11,20,rentOcc); 
  lims[1]=1;lims[2]=20;
  deleteFields("H044",lims,1);
  
  lims[1]=1;lims[2]=2;
  deleteFields("H045",lims,1);
  
  
  
    for(i=1;i<=NF;i++){
      if(whichToPrint[i]>0) printf("%8.7f\t",$i);
    }
    printf("\n");
  
  
}
