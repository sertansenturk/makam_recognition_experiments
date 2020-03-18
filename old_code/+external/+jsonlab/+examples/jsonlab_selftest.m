%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Regression Test Unit of external.jsonlab.loadjson and external.jsonlab.savejson
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:4
    fname=sprintf('example%d.json',i);
    if(exist(fname,'file')==0) break; end
    fprintf(1,'===============================================\n>> %s\n',fname);
    json=external.jsonlab.savejson('data',external.jsonlab.loadjson(fname));
    fprintf(1,'%s\n',json);
    fprintf(1,'%s\n',external.jsonlab.savejson('data',external.jsonlab.loadjson(fname),'Compact',1));
    data=external.jsonlab.loadjson(json);
    external.jsonlab.savejson('data',data,'selftest.json');
    data=external.jsonlab.loadjson('selftest.json');
end

for i=1:4
    fname=sprintf('example%d.json',i);
    if(exist(fname,'file')==0) break; end
    fprintf(1,'===============================================\n>> %s\n',fname);
    json=external.jsonlab.saveubjson('data',external.jsonlab.loadjson(fname));
    fprintf(1,'%s\n',json);
    data=external.jsonlab.loadubjson(json);
    external.jsonlab.savejson('',data);
    external.jsonlab.saveubjson('data',data,'selftest.ubj');
    data=external.jsonlab.loadubjson('selftest.ubj');
end
