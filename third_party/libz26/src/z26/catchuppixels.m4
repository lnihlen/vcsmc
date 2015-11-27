changecom(`/*',`*/')dnl
/*

	TIA pixel rendering loop for z26


 This C file gets generated from catchuppixels.m4, so please edit it there.

 z26 is Copyright 1997-2011 by John Saeger and contributors.  
 z26 is released subject to the terms and conditions of the 
 GNU General Public License Version 2 (GPL).  z26 comes with no warranty.
 Please see COPYING.TXT for details.
*/

define(handleP0,`
					s->TIA_P0_counter++;
					if(s->TIA_P0_counter == 160){
						
						s->TIA_P0_counter = s->TIA_P0_counter_reset;
						/*
							0x2000 = handle hand rendered graphics
							
							0x1000 REFP0
							0x0ff0 GRP0_new or GRP0_old value
							0x0008 show first copy of current NUSIZ0
							0x0007 s->NUSIZ0_number
						*/
						s->Pointer_Index_P0 = s->NUSIZ0_number | s->TIA_REFP0;
						if(s->TIA_P0_counter_reset == 0) s->Pointer_Index_P0 |= 0x0008;
						if(s->TIA_VDELP0) s->Pointer_Index_P0 |= s->TIA_GRP0_old;
						else s->Pointer_Index_P0 |= s->TIA_GRP0_new;
						
						s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];
						s->TIA_P0_counter_reset = 0;
					/* TODO: handle RESPM0 here */
					}')dnl
define(handleP1,`
					s->TIA_P1_counter++;
					if(s->TIA_P1_counter == 160){
						
						s->TIA_P1_counter = s->TIA_P1_counter_reset;
						/*
							0x2000 = handle hand rendered graphics
							
							0x1000 REFP1
							0x0ff0 GRP1_new or GRP1_old value
							0x0008 show first copy of current NUSIZ1
							0x0007 s->NUSIZ1_number
						*/
						s->Pointer_Index_P1 = s->NUSIZ1_number | s->TIA_REFP1;
						if(s->TIA_P1_counter_reset == 0) s->Pointer_Index_P1 |= 0x0008;
						if(s->TIA_VDELP1) s->Pointer_Index_P1 |= s->TIA_GRP1_old;
						else s->Pointer_Index_P1 |= s->TIA_GRP1_new;
						
						s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];
						s->TIA_P1_counter_reset = 0;
					/* TODO: handle RESPM1 here */
					}')dnl
define(handleM0,`
					s->TIA_M0_counter++;
					if(s->TIA_M0_counter == 160){
						
						s->TIA_M0_counter = s->TIA_M0_counter_reset;
						/*
							0x80 = handle hand rendered graphics
							
							0x40 s->TIA_ENAM0
							0x30 s->NUSIZ_M0_width
							0x08 show first copy of current NUSIZ0
							0x07 s->NUSIZ0_number
						*/
						s->Pointer_Index_M0 = s->NUSIZ0_number | s->NUSIZ_M0_width | s->TIA_ENAM0;
						if(s->TIA_M0_counter_reset == 0) s->Pointer_Index_M0 |= 0x08;
						
						s->TIA_M0_Line_Pointer = TIA_M0_Table[s->Pointer_Index_M0];
						s->TIA_M0_counter_reset = 0;
					}')dnl
define(handleM1,`
					s->TIA_M1_counter++;
					if(s->TIA_M1_counter == 160){
						
						s->TIA_M1_counter = s->TIA_M1_counter_reset;
						/*
							0x80 = handle hand rendered graphics
							
							0x40 s->TIA_ENAM1
							0x30 s->NUSIZ_M1_width
							0x08 show first copy of current NUSIZ1
							0x07 s->NUSIZ1_number
						*/
						s->Pointer_Index_M1 = s->NUSIZ1_number | s->NUSIZ_M1_width | s->TIA_ENAM1;
						if(s->TIA_M1_counter_reset == 0) s->Pointer_Index_M1 |= 0x08;
						
						s->TIA_M1_Line_Pointer = TIA_M1_Table[s->Pointer_Index_M1];
						s->TIA_M1_counter_reset = 0;
					}')dnl
define(handleBL,`
					s->TIA_BL_counter++;
					if(s->TIA_BL_counter == 160){
						
						s->TIA_BL_counter = s->TIA_BL_counter_reset;
						/*
							0x08 = handle hand rendered graphics
							
							0x04 s->TIA_ENABL_new or s->TIA_ENABL_old
							0x03 s->CTRLPF_BL_width
						*/
						s->Pointer_Index_BL = s->CTRLPF_BL_width;
						if(s->TIA_VDELBL) s->Pointer_Index_BL |= s->TIA_ENABL_old;
						else s->Pointer_Index_BL |= s->TIA_ENABL_new;
						
						s->TIA_BL_Line_Pointer = TIA_BL_Table[s->Pointer_Index_BL];
						s->TIA_BL_counter_reset = 0;
					}')dnl
define(handleH1,`
				if((s->LoopCount & 0x03) == 1){
					/* counter at H1 = HIGH */
					if(s->TIA_HMOVE_Setup == 1) s->TIA_HMOVE_Setup = 2;
					if(s->TIA_HMOVE_Latches){
						if(s->TIA_HMP0_Value == (s->TIA_HMOVE_Clock & 0x0f)) s->TIA_HMOVE_Latches &= 0x1e;
						if(s->TIA_HMP1_Value == (s->TIA_HMOVE_Clock & 0x0f)) s->TIA_HMOVE_Latches &= 0x2e;
						if(s->TIA_HMM0_Value == (s->TIA_HMOVE_Clock & 0x0f)) s->TIA_HMOVE_Latches &= 0x36;
						if(s->TIA_HMM1_Value == (s->TIA_HMOVE_Clock & 0x0f)) s->TIA_HMOVE_Latches &= 0x3a;
						if(s->TIA_HMBL_Value == (s->TIA_HMOVE_Clock & 0x0f)) s->TIA_HMOVE_Latches &= 0x3c;
					}
				}')dnl
define(handleH2,
				if((s->LoopCount & 0x03) == 3){
					/* counter at H2 = HIGH */
					s->TIA_HMOVE_DoMove = s->TIA_HMOVE_Latches;
					if(s->TIA_HMOVE_Clock < 16) s->TIA_HMOVE_Clock++;
					if(s->TIA_HMOVE_Setup == 2){
						s->TIA_HMOVE_Setup = 0;
						if(s->TIA_HMOVE_Clock == 16) s->TIA_HMOVE_Clock = 0;
							/* only reset if HMOVE isn't already in process */
						s->TIA_HMOVE_Latches = 0x3e;
							/* enable movement for all 5 objects */
					}
				})dnl
define(handlePF,`
					if((s->LoopCount & 0x03) == 0){
						if(TIA_Playfield_Pixels[(((s->LoopCount - 68) >> 2) + s->TIA_REFPF_Flag)] & s->TIA_Playfield_Bits)
							s->Current_PF_Pixel = 0x01;
						else s->Current_PF_Pixel = 0x00;
					};')dnl
define(combineObjectsOnly,`
					s->TIA_Pixel_State = s->TIA_P0_Line_Pointer[s->TIA_P0_counter]
					                | s->TIA_P1_Line_Pointer[s->TIA_P1_counter]
					                | s->TIA_M0_Line_Pointer[s->TIA_M0_counter]
					                | s->TIA_M1_Line_Pointer[s->TIA_M1_counter]
					                | s->TIA_BL_Line_Pointer[s->TIA_BL_counter];')dnl
define(combineObjectsPF,`
					s->TIA_Pixel_State = s->Current_PF_Pixel
					                | s->TIA_P0_Line_Pointer[s->TIA_P0_counter]
					                | s->TIA_P1_Line_Pointer[s->TIA_P1_counter]
					                | s->TIA_M0_Line_Pointer[s->TIA_M0_counter]
					                | s->TIA_M1_Line_Pointer[s->TIA_M1_counter]
					                | s->TIA_BL_Line_Pointer[s->TIA_BL_counter];')dnl
define(handleCollisions,`
					/* TODO: add support for for PAL colour loss */
					
					s->TIACollide |= TIA_Collision_Table[s->TIA_Pixel_State];')dnl
define(handlePriorityLine,
					/* let user disable objects */
					s->TIA_Pixel_State &= s->TIA_Mask_Objects;
					/* playfield doesn't use score colouring mode when it has display priority */
					if(s->CTRLPF_Score) *s->DisplayPointer =
						s->TIA_Colour_Table[TIA_Score_Priority_Table[(s->LoopCount - 68) / 80][s->TIA_Pixel_State]];
					else *s->DisplayPointer =
						s->TIA_Colour_Table[TIA_Priority_Table[s->CTRLPF_Priority][s->TIA_Pixel_State]];
					
					s->DisplayPointer++;)dnl
define(handlePriorityMidline,				
					/* let user disable objects */
					s->TIA_Pixel_State &= s->TIA_Mask_Objects;
					/* playfield doesn't use score colouring mode when it has display priority */
					if(s->CTRLPF_Score){
						/*
							Due to a race condition in the TIA colour encoder the last half
							pixel of the last PF quad in the left screen half will get a
							temperature dependant mix of the P0, P1 and PF colour in score mode.
							We simulate it be setting the colour of that half pixel to PF colour.
						*/
						if(s->TIA_Pixel_State == 0x01)	/* only playfield active? */
							*s->DisplayPointer =
								/* TODO: make this endian safe */
								(s->TIA_Colour_Table[P0_COLOUR] & 0x00ff) | (s->TIA_Colour_Table[PF_COLOUR] & 0xff00);
						else *s->DisplayPointer =
							s->TIA_Colour_Table[TIA_Score_Priority_Table[(s->LoopCount - 68) / 80][s->TIA_Pixel_State]];
					}else *s->DisplayPointer =
						s->TIA_Colour_Table[TIA_Priority_Table[s->CTRLPF_Priority][s->TIA_Pixel_State]];
					
					s->DisplayPointer++;)dnl
define(handleVisibleDoOutput,`
				handlePF
				combineObjectsPF
				handleP0
				handleP1
				handleM0
				handleM1
				handleBL
				handleCollisions')dnl
define(handleVisibleVBlank,`
				handlePF
				handleP0
				handleP1
				handleM0
				handleM1
				handleBL
				*s->DisplayPointer = 0;
				s->DisplayPointer++;')dnl
define(handleVisibleVBlankNoOutput,`
				handlePF
				handleP0
				handleP1
				handleM0
				handleM1
				handleBL')dnl
define(handleVisibleNoOutput,`
				handlePF
				combineObjectsPF
				handleP0
				handleP1
				handleM0
				handleM1
				handleBL
				handleCollisions')dnl
define(handleInvisibleCollision,`
					if(s->TIA_HMOVE_DoMove){
						combineObjectsOnly
						handleCollisions
						if(s->TIA_HMOVE_DoMove & 0x20){					
							handleP0
						}
						if(s->TIA_HMOVE_DoMove & 0x10){					
							handleP1
						}
						if(s->TIA_HMOVE_DoMove & 0x08){					
							handleM0
						}
						if(s->TIA_HMOVE_DoMove & 0x04){					
							handleM1
						}
						if(s->TIA_HMOVE_DoMove & 0x02){					
							handleBL
						}
						s->TIA_HMOVE_DoMove = 0;
					}	')dnl
define(handleInvisibleVBlank,`
					if(s->TIA_HMOVE_DoMove){
						if(s->TIA_HMOVE_DoMove & 0x20){					
							handleP0
						}
						if(s->TIA_HMOVE_DoMove & 0x10){					
							handleP1
						}
						if(s->TIA_HMOVE_DoMove & 0x08){					
							handleM0
						}
						if(s->TIA_HMOVE_DoMove & 0x04){					
							handleM1
						}
						if(s->TIA_HMOVE_DoMove & 0x02){					
							handleBL
						}
						s->TIA_HMOVE_DoMove = 0;
					}	')dnl
define(renderBlank,`
			*s->DisplayPointer = 0;
			s->DisplayPointer++;')dnl

define(renderLoop,

		 	for(s->CountLoop = s->TIA_Last_Pixel; s->CountLoop < ((s->RClock * 3) + s->TIA_Delayed_Write); s->CountLoop++){
		 		s->LoopCount = s->CountLoop;
				if(s->LoopCount > 227) s->LoopCount -= 228;
		
				handleH1
				handleH2

				if(s->LoopCount > 75){
					
					if(s->LoopCount == 147){
						
					/*
						we're at the center of the displayed line here
						-> queue a sound byte
						-> test REFPF bit
						-> fix half a pixel of last PF pixel in score mode
					*/
			
						$1
						$4 
		
						/* The PF reflect bit gets only checked at center screen. */
						if(s->CTRLPF_PF_Reflect) s->TIA_REFPF_Flag = 40;
						else s->TIA_REFPF_Flag = 0;
		
					}else{
		
						$1
						$5
					}	
				}else if(s->LoopCount < 68){
		
					$2
						
				}else if(s->TIA_Display_HBlank){
		
					$3
					if(s->LoopCount == 75) s->TIA_Display_HBlank = 0;
		
					$2
		
				}else{
		
					$1
					$5
				}	
			}
			s->TIA_Last_Pixel = (s->RClock * 3) + s->TIA_Delayed_Write;)dnl

	if(s->TIA_Do_Output){
		if(s->TIA_VBLANK){
			renderLoop(handleVisibleVBlank,handleInvisibleVBlank,renderBlank)
		}else{
			renderLoop(handleVisibleDoOutput,handleInvisibleCollision,renderBlank,handlePriorityMidline,handlePriorityLine)
		}
	}else{
		if(s->TIA_VBLANK){
			renderLoop(handleVisibleVBlankNoOutput,handleInvisibleVBlank)
		}else{
			renderLoop(handleVisibleNoOutput,handleInvisibleCollision)
		}
	}
