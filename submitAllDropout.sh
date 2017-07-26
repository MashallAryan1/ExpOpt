#!/bin/bash
#
#func_id =  eval(argv[0])
#dim =  eval(argv[1])
#instance = eval(argv[2])
#struc_index = eval(argv[3]) # [0,1]
#lr_index = eval(argv[4])
#dr_index = eval(argv[5])

idim=$1
i=0
        for lr_index in {0,1,2}
            do
                for dr_index in {0,1,2,3}
                    do
                        for struc_index in {0,1}
                           do
                               for funcId in {1,}
                                    do                                                                                    
                                         i=$((i+1))
                                         echo ${i}
                                         echo $funcId $instance  $lr_index $dr_index $struc_index
	   	  	  	   	 qsub -t 1-30 submit-cocoex-01.sh $funcId $idim   $struc_index  $lr_index $dr_index                                        
                                    done
                           done
                    done
            done

