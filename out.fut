let {i64 res_10542} =
  i32_1904(24i32)
let {i64 res_10543} =
  i32_1904(12i32)
let {i64 res_10544} =
  i32_1904(256i32)
let {f32 res_10545} =
  /_3222(1.0f32, 0.0f32)
let {f32 res_10546} = fsub32(0.0f32, res_10545)
let {f32 res_10547} =
  f64_3248(2.718281828459045f64)
let {i32 res_10548} =
  i32_2549(0i32)
let {f32 res_10549} =
  f64_3248(3.141592653589793f64)

fun {i64} i32_1904 (i32 x_10550) = {
  let {i64 res_10551} = sext i32 x_10550 to i64
  in {res_10551}
}

fun {f32} max_3281 (f32 x_10552, f32 y_10553) = {
  let {f32 res_10554} = fmax32(x_10552, y_10553)
  in {res_10554}
}

fun {f32} /_3222 (f32 x_10555, f32 y_10556) = {
  let {f32 res_10557} = fdiv32(x_10555, y_10556)
  in {res_10557}
}

fun {bool} lifted_0_reduce_7824 (bool nameless_10558, bool op_10559) = {
  {op_10559}
}

fun {f32, bool} lifted_1_reduce_7825 (bool op_10560, f32 ne_10561) = {
  {ne_10561, op_10560}
}

fun {f32} lifted_0_op_7828 (bool nameless_10562, f32 x_10563) = {
  {x_10563}
}

fun {f32} lifted_1_op_7829 (f32 x_10564, f32 y_10565) = {
  let {f32 res_10566} = fmax32(x_10564, y_10565)
  in {res_10566}
}

fun {f32} lifted_2_reduce_7830 (i64 n_10567, f32 ne_10568, bool op_10569,
                                [n_10567]f32 as_10570) = {
  let {f32 res_10571} =
    redomap(n_10567,
            {fn {f32} (f32 x_10572, f32 x_10573) =>
               let {f32 lifted_1_op_arg_10574} =
                 lifted_0_op_7828(op_10569, x_10572)
               let {f32 res_10575} =
                 lifted_1_op_7829(lifted_1_op_arg_10574, x_10573)
               in {res_10575},
             {ne_10568}},
            fn {f32} (f32 x_10576) =>
              {x_10576},
            as_10570)
  in {res_10571}
}

fun {f32} maximum_7678 (i64 n₀_10577, [n₀_10577]f32 as_10578) = {
  let {bool lifted_1_reduce_arg_10579} =
    lifted_0_reduce_7824(true, true)
  let {f32 lifted_2_reduce_arg_10580, bool lifted_2_reduce_arg_10581} =
    lifted_1_reduce_7825(lifted_1_reduce_arg_10579, res_10546)
  let {f32 res_10582} =
    lifted_2_reduce_7830(n₀_10577, lifted_2_reduce_arg_10580,
                         lifted_2_reduce_arg_10581, as_10578)
  in {res_10582}
}

fun {[n_10583]f32, [n_10583]i64} zip_7680 (i64 n_10583, [n_10583]f32 as_10584,
                                           [n_10583]i64 bs_10585) = {
  let {i64 ret₂_10586} = n_10583
  in {as_10584, bs_10585}
}

fun {[n_10587]f32, [n_10587]i64} zip2_7681 (i64 n_10587, [n_10587]f32 as_10588,
                                            [n_10587]i64 bs_10589) = {
  -- res_10590 aliases as_10588, bs_10589
  -- res_10591 aliases as_10588, bs_10589
  let {[n_10587]f32 res_10590, [n_10587]i64 res_10591} =
    zip_7680(n_10587, as_10588, bs_10589)
  in {res_10590, res_10591}
}

fun {f32} i64_3244 (i64 x_10592) = {
  let {f32 res_10593} = sitofp i64 x_10592 to f32
  in {res_10593}
}

fun {bool} lifted_0_map_7831 (bool nameless_10594, bool f_10595) = {
  {f_10595}
}

fun {f32} lifted_0_f_7833 (bool nameless_10596, i64 x_10597) = {
  let {f32 res_10598} = sitofp i64 x_10597 to f32
  in {res_10598}
}

fun {*[n_10599]f32} lifted_1_map_7834 (i64 n_10599, bool f_10600,
                                       [n_10599]i64 as_10601) = {
  let {[n_10599]f32 res_10602} =
    map(n_10599,
        fn {f32} (i64 x_10603) =>
          let {f32 res_10604} =
            lifted_0_f_7833(f_10600, x_10603)
          in {res_10604},
        as_10601)
  let {i64 ret₂_10605} = n_10599
  in {res_10602}
}

fun {f32} lifted_0_map_7835 (bool nameless_10606, f32 f_10607) = {
  {f_10607}
}

fun {f32} lifted_0_f_7837 (f32 swap_term_10608, f32 x_10609) = {
  let {f32 res_10610} = fmul32(x_10609, swap_term_10608)
  in {res_10610}
}

fun {*[n_10611]f32} lifted_1_map_7838 (i64 n_10611, f32 f_10612,
                                       [n_10611]f32 as_10613) = {
  let {[n_10611]f32 res_10614} =
    map(n_10611,
        fn {f32} (f32 x_10615) =>
          let {f32 res_10616} =
            lifted_0_f_7837(f_10612, x_10615)
          in {res_10616},
        as_10613)
  let {i64 ret₂_10617} = n_10611
  in {res_10614}
}

fun {*[?0]f32} gen_payment_dates_7233 (i64 swap_payments_10618,
                                       f32 swap_term_10619) = {
  let {i64 range_end_10620} = sub64(swap_payments_10618, 1i64)
  let {i64 subtracted_step_10621} = sub64(1i64, 0i64)
  let {bool step_zero_10622} = eq_i64(0i64, 1i64)
  let {i64 s_sign_10623} = ssignum64 subtracted_step_10621
  let {bool bounds_invalid_downwards_10624} = sle64(0i64, range_end_10620)
  let {bool bounds_invalid_upwards_10625} = slt64(range_end_10620, 0i64)
  let {bool downwards_10626} = eq_i64(s_sign_10623, -1i64)
  let {i64 distance_downwards_exclusive_10627} = sub64(0i64, range_end_10620)
  let {i64 distance_upwards_exclusive_10628} = sub64(range_end_10620, 0i64)
  let {bool bounds_invalid_10629} =
    -- Branch returns: {bool}
    if downwards_10626
    then {bounds_invalid_downwards_10624} else {bounds_invalid_upwards_10625}
  let {i64 distance_exclusive_10630} =
    -- Branch returns: {i64}
    if downwards_10626
    then {distance_downwards_exclusive_10627} else {distance_upwards_exclusive_10628}
  let {i64 distance_10631} = add64(distance_exclusive_10630, 1i64)
  let {bool step_invalid_10632} = logor(false, step_zero_10622)
  let {bool range_invalid_10633} =
    logor(step_invalid_10632, bounds_invalid_10629)
  let {bool valid_10634} = not range_invalid_10633
  let {cert range_valid_c_10635} =
    assert(valid_10634, "Range ", 0i64, "..", 1i64, "...", range_end_10620,
                        " is invalid.", "cva.fut:54:29-52")
  let {i64 pos_step_10636} = mul64(subtracted_step_10621, s_sign_10623)
  let {i64 num_elems_10637} =
    <range_valid_c_10635>
    sdiv_up64(distance_10631, pos_step_10636)
  let {[num_elems_10637]i64 lifted_1_map_arg_10638} = iota64(num_elems_10637,
                                                             0i64,
                                                             subtracted_step_10621)
  let {i64 range_dim₇_10639} = num_elems_10637
  let {bool lifted_1_map_arg_10640} =
    lifted_0_map_7831(true, true)
  let {[num_elems_10637]f32 seq_10641} =
    lifted_1_map_7834(num_elems_10637, lifted_1_map_arg_10640,
                      lifted_1_map_arg_10638)
  -- seq_10642 aliases seq_10641
  let {[num_elems_10637]f32 seq_10642} = seq_10641
  let {f32 lifted_1_map_arg_10643} =
    lifted_0_map_7835(true, swap_term_10619)
  let {[num_elems_10637]f32 res_10644} =
    lifted_1_map_7838(num_elems_10637, lifted_1_map_arg_10643, seq_10642)
  in {num_elems_10637, res_10644}
}

fun {f32} f64_3248 (f64 x_10645) = {
  let {f32 res_10646} = fpconv f64 x_10645 to f32
  in {res_10646}
}

fun {f32} exp_7109 (f32 a_10647) = {
  let {f32 res_10648} = fpow32(res_10547, a_10647)
  in {res_10648}
}

fun {f32} bondprice_7156 (f32 a_10649, f32 b_10650, f32 deltat_10651,
                          f32 r0_10652, f32 sigma_10653, f32 r_10654,
                          f32 t_10655, f32 T_10656) = {
  let {f32 y_10657} = fsub32(T_10656, t_10655)
  let {f32 negate_arg_10658} = fmul32(a_10649, y_10657)
  let {f32 exp_arg_10659} = fsub32(0.0f32, negate_arg_10658)
  let {f32 y_10660} =
    exp_7109(exp_arg_10659)
  let {f32 x_10661} = fsub32(1.0f32, y_10660)
  let {f32 B_10662} = fdiv32(x_10661, a_10649)
  let {f32 B_10663} = B_10662
  let {f32 x_10664} = fsub32(B_10663, T_10656)
  let {f32 x_10665} = fadd32(x_10664, t_10655)
  let {f32 x_10666} = fpow32(a_10649, 2.0f32)
  let {f32 x_10667} = fmul32(x_10666, b_10650)
  let {f32 x_10668} = fpow32(sigma_10653, 2.0f32)
  let {f32 y_10669} = fdiv32(x_10668, 2.0f32)
  let {f32 y_10670} = fsub32(x_10667, y_10669)
  let {f32 x_10671} = fmul32(x_10665, y_10670)
  let {f32 y_10672} = fpow32(a_10649, 2.0f32)
  let {f32 A1_10673} = fdiv32(x_10671, y_10672)
  let {f32 A1_10674} = A1_10673
  let {f32 x_10675} = fpow32(sigma_10653, 2.0f32)
  let {f32 y_10676} = fpow32(B_10663, 2.0f32)
  let {f32 x_10677} = fmul32(x_10675, y_10676)
  let {f32 y_10678} = fmul32(4.0f32, a_10649)
  let {f32 A2_10679} = fdiv32(x_10677, y_10678)
  let {f32 A2_10680} = A2_10679
  let {f32 exp_arg_10681} = fsub32(A1_10674, A2_10680)
  let {f32 A_10682} =
    exp_7109(exp_arg_10681)
  let {f32 A_10683} = A_10682
  let {f32 negate_arg_10684} = fmul32(B_10663, r_10654)
  let {f32 exp_arg_10685} = fsub32(0.0f32, negate_arg_10684)
  let {f32 y_10686} =
    exp_7109(exp_arg_10685)
  let {f32 res_10687} = fmul32(A_10683, y_10686)
  in {res_10687}
}

fun {bool} lifted_0_reduce_7841 (bool nameless_10688, bool op_10689) = {
  {op_10689}
}

fun {f32, bool} lifted_1_reduce_7842 (bool op_10690, f32 ne_10691) = {
  {ne_10691, op_10690}
}

fun {f32, f32, f32, f32, f32} lifted_0_map_7843 (bool nameless_10692,
                                                 f32 f_10693, f32 f_10694,
                                                 f32 f_10695, f32 f_10696,
                                                 f32 f_10697) = {
  {f_10693, f_10694, f_10695, f_10696, f_10697}
}

fun {f32} lifted_0_f_7845 (f32 a_10698, f32 b_10699, f32 deltat_10700,
                           f32 r0_10701, f32 sigma_10702, f32 x_10703) = {
  let {f32 res_10704} =
    bondprice_7156(a_10698, b_10699, deltat_10700, r0_10701, sigma_10702,
                   r0_10701, 0.0f32, x_10703)
  in {res_10704}
}

fun {*[n_10705]f32} lifted_1_map_7846 (i64 n_10705, f32 f_10706, f32 f_10707,
                                       f32 f_10708, f32 f_10709, f32 f_10710,
                                       [n_10705]f32 as_10711) = {
  let {[n_10705]f32 res_10712} =
    map(n_10705,
        fn {f32} (f32 x_10713) =>
          let {f32 res_10714} =
            lifted_0_f_7845(f_10706, f_10707, f_10708, f_10709, f_10710,
                            x_10713)
          in {res_10714},
        as_10711)
  let {i64 ret₂_10715} = n_10705
  in {res_10712}
}

fun {f32} lifted_0_op_7849 (bool nameless_10716, f32 x_10717) = {
  {x_10717}
}

fun {f32} lifted_1_op_7850 (f32 x_10718, f32 x_10719) = {
  let {f32 res_10720} = fadd32(x_10718, x_10719)
  in {res_10720}
}

fun {f32} lifted_2_reduce_7851 (i64 n_10721, f32 ne_10722, bool op_10723,
                                [n_10721]f32 as_10724) = {
  let {f32 res_10725} =
    redomap(n_10721,
            {fn {f32} (f32 x_10726, f32 x_10727) =>
               let {f32 lifted_1_op_arg_10728} =
                 lifted_0_op_7849(op_10723, x_10726)
               let {f32 res_10729} =
                 lifted_1_op_7850(lifted_1_op_arg_10728, x_10727)
               in {res_10729},
             {ne_10722}},
            fn {f32} (f32 x_10730) =>
              {x_10730},
            as_10724)
  in {res_10725}
}

fun {f32} set_fixed_rate_7400 (f32 swap_term_10731, i64 swap_payments_10732,
                               f32 a_10733, f32 b_10734, f32 deltat_10735,
                               f32 r0_10736, f32 sigma_10737) = {
  let {i64 size_10738;
       [size_10738]f32 payment_dates_10739} =
    gen_payment_dates_7233(swap_payments_10732, swap_term_10731)
  let {i64 ret₀_10740} = size_10738
  -- payment_dates_10741 aliases payment_dates_10739
  let {[size_10738]f32 payment_dates_10741} = payment_dates_10739
  let {i64 w_minus_1_10742} = sub64(size_10738, 1i64)
  let {cert index_certs_10743} =
    assert(true, "Index [] out of bounds for array of shape [", size_10738,
                 "].", "cva.fut:103:48-66")
  -- indexed_10744 aliases payment_dates_10741
  let {[size_10738]f32 indexed_10744} =
    <index_certs_10743>
    payment_dates_10741[w_minus_1_10742:+size_10738*-1i64]
  let {i64 to_i64_10745} = sext i32 0i32 to i64
  let {bool x_10746} = sle64(0i64, to_i64_10745)
  let {bool y_10747} = slt64(to_i64_10745, size_10738)
  let {bool bounds_check_10748} = logand(x_10746, y_10747)
  let {cert index_certs_10749} =
    assert(bounds_check_10748, "Index [", to_i64_10745,
                               "] out of bounds for array of shape [",
                               size_10738, "].", "cva.fut:103:47-70")
  let {f32 bondprice_arg_10750} =
    <index_certs_10749>
    indexed_10744[to_i64_10745]
  let {f32 leg1_10751} =
    bondprice_7156(a_10733, b_10734, deltat_10735, r0_10736, sigma_10737,
                   r0_10736, 0.0f32, bondprice_arg_10750)
  let {f32 leg1_10752} = leg1_10751
  let {i64 s_sign_10753} = ssignum64 1i64
  let {bool backwards_10754} = eq_i64(s_sign_10753, -1i64)
  let {i64 w_minus_1_10755} = sub64(size_10738, 1i64)
  let {i64 j_def_10756} =
    -- Branch returns: {i64}
    if backwards_10754
    then {-1i64} else {size_10738}
  let {i64 j_m_i_10757} = sub64(j_def_10756, 1i64)
  let {i64 y_10758} = ssignum64 1i64
  let {i64 y_10759} = sub64(1i64, y_10758)
  let {i64 x_10760} = add64(j_m_i_10757, y_10759)
  let {i64 n_10761} = squot64(x_10760, 1i64)
  let {bool empty_slice_10762} = eq_i64(n_10761, 0i64)
  let {i64 m_10763} = sub64(n_10761, 1i64)
  let {i64 m_t_s_10764} = mul64(m_10763, 1i64)
  let {i64 i_p_m_t_s_10765} = add64(1i64, m_t_s_10764)
  let {bool zero_leq_i_p_m_t_s_10766} = sle64(0i64, i_p_m_t_s_10765)
  let {bool i_p_m_t_s_leq_w_10767} = sle64(i_p_m_t_s_10765, size_10738)
  let {bool i_p_m_t_s_leq_w_10768} = slt64(i_p_m_t_s_10765, size_10738)
  let {bool zero_lte_i_10769} = sle64(0i64, 1i64)
  let {bool i_lte_j_10770} = sle64(1i64, j_def_10756)
  let {bool y_10771} = logand(i_p_m_t_s_leq_w_10768, zero_lte_i_10769)
  let {bool y_10772} = logand(zero_leq_i_p_m_t_s_10766, y_10771)
  let {bool y_10773} = logand(i_lte_j_10770, y_10772)
  let {bool forwards_ok_10774} = logand(zero_lte_i_10769, y_10773)
  let {bool negone_lte_j_10775} = sle64(-1i64, j_def_10756)
  let {bool j_lte_i_10776} = sle64(j_def_10756, 1i64)
  let {bool y_10777} = logand(i_p_m_t_s_leq_w_10767, negone_lte_j_10775)
  let {bool y_10778} = logand(zero_leq_i_p_m_t_s_10766, y_10777)
  let {bool y_10779} = logand(j_lte_i_10776, y_10778)
  let {bool backwards_ok_10780} = logand(negone_lte_j_10775, y_10779)
  let {bool slice_ok_10781} =
    -- Branch returns: {bool}
    if backwards_10754
    then {backwards_ok_10780} else {forwards_ok_10774}
  let {bool ok_or_empty_10782} = logor(empty_slice_10762, slice_ok_10781)
  let {cert index_certs_10783} =
    assert(ok_or_empty_10782, "Index [", 1i64,
                              ":] out of bounds for array of shape [",
                              size_10738, "].", "cva.fut:104:74-90")
  -- lifted_1_map_arg_10784 aliases payment_dates_10741
  let {[n_10761]f32 lifted_1_map_arg_10784} =
    <index_certs_10783>
    payment_dates_10741[1i64:+n_10761*1i64]
  let {i64 argdim₁₉_10785} = n_10761
  let {f32 lifted_1_map_arg_10786, f32 lifted_1_map_arg_10787,
       f32 lifted_1_map_arg_10788, f32 lifted_1_map_arg_10789,
       f32 lifted_1_map_arg_10790} =
    lifted_0_map_7843(true, a_10733, b_10734, deltat_10735, r0_10736,
                      sigma_10737)
  let {[n_10761]f32 lifted_2_reduce_arg_10791} =
    lifted_1_map_7846(n_10761, lifted_1_map_arg_10786, lifted_1_map_arg_10787,
                      lifted_1_map_arg_10788, lifted_1_map_arg_10789,
                      lifted_1_map_arg_10790, lifted_1_map_arg_10784)
  let {bool lifted_1_reduce_arg_10792} =
    lifted_0_reduce_7841(true, true)
  let {f32 lifted_2_reduce_arg_10793, bool lifted_2_reduce_arg_10794} =
    lifted_1_reduce_7842(lifted_1_reduce_arg_10792, 0.0f32)
  let {f32 sum_10795} =
    lifted_2_reduce_7851(n_10761, lifted_2_reduce_arg_10793,
                         lifted_2_reduce_arg_10794, lifted_2_reduce_arg_10791)
  let {f32 sum_10796} = sum_10795
  let {f32 x_10797} = fsub32(1.0f32, leg1_10752)
  let {f32 y_10798} = fmul32(sum_10796, swap_term_10731)
  let {f32 res_10799} = fdiv32(x_10797, y_10798)
  in {res_10799}
}

fun {*[n_10800]i64} iota_3662 (i64 n_10800) = {
  let {i64 subtracted_step_10801} = sub64(1i64, 0i64)
  let {bool step_zero_10802} = eq_i64(0i64, 1i64)
  let {i64 s_sign_10803} = ssignum64 subtracted_step_10801
  let {bool bounds_invalid_downwards_10804} = sle64(0i64, n_10800)
  let {bool bounds_invalid_upwards_10805} = slt64(n_10800, 0i64)
  let {bool step_wrong_dir_10806} = eq_i64(s_sign_10803, -1i64)
  let {i64 distance_10807} = sub64(n_10800, 0i64)
  let {bool step_invalid_10808} = logor(step_wrong_dir_10806, step_zero_10802)
  let {bool range_invalid_10809} =
    logor(step_invalid_10808, bounds_invalid_upwards_10805)
  let {bool valid_10810} = not range_invalid_10809
  let {cert range_valid_c_10811} =
    assert(valid_10810, "Range ", 0i64, "..", 1i64, "..<", n_10800,
                        " is invalid.", "/prelude/array.fut:60:3-10")
  let {i64 pos_step_10812} = mul64(subtracted_step_10801, s_sign_10803)
  let {i64 num_elems_10813} =
    <range_valid_c_10811>
    sdiv_up64(distance_10807, pos_step_10812)
  let {[num_elems_10813]i64 res_10814} = iota64(num_elems_10813, 0i64,
                                                subtracted_step_10801)
  let {bool dim_match_10815} = eq_i64(n_10800, num_elems_10813)
  let {cert empty_or_match_cert_10816} =
    assert(dim_match_10815,
           "Function return value does not match shape of type *[", n_10800,
           "]i64", "/prelude/array.fut:59:1-60:10")
  -- result_proper_shape_10817 aliases res_10814
  let {[n_10800]i64 result_proper_shape_10817} =
    <empty_or_match_cert_10816>
    reshape((~n_10800), res_10814)
  in {result_proper_shape_10817}
}

fun {*[n_10818]i64} indices_7702 (i64 n_10818, [n_10818]f32 nameless_10819) = {
  let {[n_10818]i64 res_10820} =
    iota_3662(n_10818)
  in {res_10820}
}

fun {f32} lifted_0_map_7852 (bool nameless_10821, f32 f_10822) = {
  {f_10822}
}

fun {bool} lifted_0_map_7853 (bool nameless_10823, bool f_10824) = {
  {f_10824}
}

fun {f32} lifted_0_f_7855 (bool nameless_10825, i64 x_10826) = {
  let {f32 res_10827} = sitofp i64 x_10826 to f32
  in {res_10827}
}

fun {*[n_10828]f32} lifted_1_map_7856 (i64 n_10828, bool f_10829,
                                       [n_10828]i64 as_10830) = {
  let {[n_10828]f32 res_10831} =
    map(n_10828,
        fn {f32} (i64 x_10832) =>
          let {f32 res_10833} =
            lifted_0_f_7855(f_10829, x_10832)
          in {res_10833},
        as_10830)
  let {i64 ret₂_10834} = n_10828
  in {res_10831}
}

fun {f32} lifted_0_f_7858 (f32 sims_per_year_10835, f32 x_10836) = {
  let {f32 res_10837} = fdiv32(x_10836, sims_per_year_10835)
  in {res_10837}
}

fun {*[n_10838]f32} lifted_1_map_7859 (i64 n_10838, f32 f_10839,
                                       [n_10838]f32 as_10840) = {
  let {[n_10838]f32 res_10841} =
    map(n_10838,
        fn {f32} (f32 x_10842) =>
          let {f32 res_10843} =
            lifted_0_f_7858(f_10839, x_10842)
          in {res_10843},
        as_10840)
  let {i64 ret₂_10844} = n_10838
  in {res_10841}
}

fun {*[steps_10845]f32} gen_times_7248 (i64 steps_10845, f32 years_10846) = {
  let {f32 x_10847} =
    i64_3244(steps_10845)
  let {f32 sims_per_year_10848} = fdiv32(x_10847, years_10846)
  let {f32 sims_per_year_10849} = sims_per_year_10848
  let {i64 subtracted_step_10850} = sub64(2i64, 1i64)
  let {bool step_zero_10851} = eq_i64(1i64, 2i64)
  let {i64 s_sign_10852} = ssignum64 subtracted_step_10850
  let {bool bounds_invalid_downwards_10853} = sle64(1i64, steps_10845)
  let {bool bounds_invalid_upwards_10854} = slt64(steps_10845, 1i64)
  let {bool downwards_10855} = eq_i64(s_sign_10852, -1i64)
  let {i64 distance_downwards_exclusive_10856} = sub64(1i64, steps_10845)
  let {i64 distance_upwards_exclusive_10857} = sub64(steps_10845, 1i64)
  let {bool bounds_invalid_10858} =
    -- Branch returns: {bool}
    if downwards_10855
    then {bounds_invalid_downwards_10853} else {bounds_invalid_upwards_10854}
  let {i64 distance_exclusive_10859} =
    -- Branch returns: {i64}
    if downwards_10855
    then {distance_downwards_exclusive_10856} else {distance_upwards_exclusive_10857}
  let {i64 distance_10860} = add64(distance_exclusive_10859, 1i64)
  let {bool step_invalid_10861} = logor(false, step_zero_10851)
  let {bool range_invalid_10862} =
    logor(step_invalid_10861, bounds_invalid_10858)
  let {bool valid_10863} = not range_invalid_10862
  let {cert range_valid_c_10864} =
    assert(valid_10863, "Range ", 1i64, "..", 2i64, "...", steps_10845,
                        " is invalid.", "cva.fut:60:56-67")
  let {i64 pos_step_10865} = mul64(subtracted_step_10850, s_sign_10852)
  let {i64 num_elems_10866} =
    <range_valid_c_10864>
    sdiv_up64(distance_10860, pos_step_10865)
  let {[num_elems_10866]i64 lifted_1_map_arg_10867} = iota64(num_elems_10866,
                                                             1i64,
                                                             subtracted_step_10850)
  let {bool lifted_1_map_arg_10868} =
    lifted_0_map_7853(true, true)
  let {[num_elems_10866]f32 lifted_1_map_arg_10869} =
    lifted_1_map_7856(num_elems_10866, lifted_1_map_arg_10868,
                      lifted_1_map_arg_10867)
  let {f32 lifted_1_map_arg_10870} =
    lifted_0_map_7852(true, sims_per_year_10849)
  let {[num_elems_10866]f32 sim_times_10871} =
    lifted_1_map_7859(num_elems_10866, lifted_1_map_arg_10870,
                      lifted_1_map_arg_10869)
  -- sim_times_10872 aliases sim_times_10871
  let {[num_elems_10866]f32 sim_times_10872} = sim_times_10871
  let {bool dim_match_10873} = eq_i64(steps_10845, num_elems_10866)
  let {cert empty_or_match_cert_10874} =
    assert(dim_match_10873,
           "Function return value does not match shape of declared return type.",
           "cva.fut:58:1-61:16")
  -- result_proper_shape_10875 aliases sim_times_10872
  let {[steps_10845]f32 result_proper_shape_10875} =
    <empty_or_match_cert_10874>
    reshape((~steps_10845), sim_times_10872)
  in {result_proper_shape_10875}
}

fun {i32} unsign_2491 (i32 x_10876) = {
  let {i32 res_10877} = zext i32 x_10876 to i32
  in {res_10877}
}

fun {i32} sign_2489 (i32 x_10878) = {
  let {i32 res_10879} = zext i32 x_10878 to i32
  in {res_10879}
}

fun {i32} ^_2524 (i32 x_10880, i32 y_10881) = {
  let {i32 x_10882} =
    sign_2489(x_10880)
  let {i32 y_10883} =
    sign_2489(y_10881)
  let {i32 unsign_arg_10884} = xor32(x_10882, y_10883)
  let {i32 res_10885} =
    unsign_2491(unsign_arg_10884)
  in {res_10885}
}

fun {i32} >>_2532 (i32 x_10886, i32 y_10887) = {
  let {i32 x_10888} =
    sign_2489(x_10886)
  let {i32 y_10889} =
    sign_2489(y_10887)
  let {i32 unsign_arg_10890} = ashr32(x_10888, y_10889)
  let {i32 res_10891} =
    unsign_2491(unsign_arg_10890)
  in {res_10891}
}

fun {i32} i32_2549 (i32 x_10892) = {
  let {i32 unsign_arg_10893} = zext i32 x_10892 to i32
  let {i32 res_10894} =
    unsign_2491(unsign_arg_10893)
  in {res_10894}
}

fun {i32} %%_2515 (i32 x_10895, i32 y_10896) = {
  let {i32 x_10897} =
    sign_2489(x_10895)
  let {i32 y_10898} =
    sign_2489(y_10896)
  let {i32 unsign_arg_10899} = umod32(x_10897, y_10898)
  let {i32 res_10900} =
    unsign_2491(unsign_arg_10899)
  in {res_10900}
}

fun {i32} +_2494 (i32 x_10901, i32 y_10902) = {
  let {i32 x_10903} =
    sign_2489(x_10901)
  let {i32 y_10904} =
    sign_2489(y_10902)
  let {i32 unsign_arg_10905} = add32(x_10903, y_10904)
  let {i32 res_10906} =
    unsign_2491(unsign_arg_10905)
  in {res_10906}
}

fun {i32} *_2500 (i32 x_10907, i32 y_10908) = {
  let {i32 x_10909} =
    sign_2489(x_10907)
  let {i32 y_10910} =
    sign_2489(y_10908)
  let {i32 unsign_arg_10911} = mul32(x_10909, y_10910)
  let {i32 res_10912} =
    unsign_2491(unsign_arg_10911)
  in {res_10912}
}

fun {i32, i32} rand_7573 (i32 x_10913) = {
  let {i32 +_arg_10914} =
    *_2500(48271i32, x_10913)
  let {i32 %%_arg_10915} =
    +_2494(+_arg_10914, 0i32)
  let {i32 rng'_10916} =
    %%_2515(%%_arg_10915, 2147483647i32)
  let {i32 rng'_10917} = rng'_10916
  in {rng'_10917, rng'_10917}
}

fun {i32} u32_1706 (i32 x_10918) = {
  let {i32 x_10919} = zext i32 x_10918 to i32
  let {i32 res_10920} = zext i32 x_10919 to i32
  in {res_10920}
}

fun {i32} u32_2541 (i32 x_10921) = {
  let {i32 unsign_arg_10922} =
    u32_1706(x_10921)
  let {i32 res_10923} =
    unsign_2491(unsign_arg_10922)
  in {res_10923}
}

fun {i32} rng_from_seed_7704 (i64 n_10924, [n_10924]i32 seed_10925) = {
  let {i32 seed'_10926} =
    loop {i32 seed'_10928} = {1i32}
    for i_10927:i64 < n_10924 do {
      let {bool x_10929} = sle64(0i64, i_10927)
      let {bool y_10930} = slt64(i_10927, n_10924)
      let {bool bounds_check_10931} = logand(x_10929, y_10930)
      let {cert index_certs_10932} =
        assert(bounds_check_10931, "Index [", i_10927,
                                   "] out of bounds for array of shape [",
                                   n_10924, "].",
               "lib/github.com/diku-dk/cpprandom/random.fut:169:19-25")
      let {i32 i32_arg_10933} =
        <index_certs_10932>
        seed_10925[i_10927]
      let {i32 ^_arg_10934} =
        i32_2549(i32_arg_10933)
      let {i32 ^_arg_10935} =
        ^_2524(^_arg_10934, 5461i32)
      let {i32 ^_arg_10936} =
        >>_2532(seed'_10928, 16i32)
      let {i32 ^_arg_10937} =
        ^_2524(^_arg_10936, seed'_10928)
      let {i32 loopres_10938} =
        ^_2524(^_arg_10937, ^_arg_10935)
      in {loopres_10938}
    }
  let {i32 seed'_10939} = seed'_10926
  let {i32 rand_arg_10940} =
    u32_2541(seed'_10939)
  let {i32 res_10941, i32 res_10942} =
    rand_7573(rand_arg_10940)
  in {res_10941}
}

fun {bool, i64} tabulate_7707 (i64 n_10943) = {
  {true, n_10943}
}

fun {i32} hash_3888 (i32 x_10944) = {
  let {i32 x_10945} =
    i32_2549(x_10944)
  let {i32 x_10946} = x_10945
  let {i32 x_10947} = lshr32(x_10946, 16i32)
  let {i32 x_10948} = xor32(x_10947, x_10946)
  let {i32 x_10949} = mul32(x_10948, 73244475i32)
  let {i32 x_10950} = x_10949
  let {i32 x_10951} = lshr32(x_10950, 16i32)
  let {i32 x_10952} = xor32(x_10951, x_10950)
  let {i32 x_10953} = mul32(x_10952, 73244475i32)
  let {i32 x_10954} = x_10953
  let {i32 x_10955} = lshr32(x_10954, 16i32)
  let {i32 x_10956} = xor32(x_10955, x_10954)
  let {i32 x_10957} = x_10956
  let {i32 res_10958} =
    u32_1706(x_10957)
  in {res_10958}
}

fun {i32} i64_1700 (i64 x_10959) = {
  let {i32 res_10960} = sext i64 x_10959 to i32
  in {res_10960}
}

fun {i32, bool} lifted_0_map1_7860 (bool map_10961, i32 f_10962) = {
  {f_10962, map_10961}
}

fun {i32} lifted_0_map_7861 (bool nameless_10963, i32 f_10964) = {
  {f_10964}
}

fun {i32} lifted_0_f_7863 (i32 x_10965, i64 i_10966) = {
  let {i32 hash_arg_10967} =
    i64_1700(i_10966)
  let {i32 i32_arg_10968} =
    hash_3888(hash_arg_10967)
  let {i32 ^_arg_10969} =
    i32_2549(i32_arg_10968)
  let {i32 res_10970} =
    ^_2524(x_10965, ^_arg_10969)
  in {res_10970}
}

fun {*[n_10971]i32} lifted_1_map_7864 (i64 n_10971, i32 f_10972,
                                       [n_10971]i64 as_10973) = {
  let {[n_10971]i32 res_10974} =
    map(n_10971,
        fn {i32} (i64 x_10975) =>
          let {i32 res_10976} =
            lifted_0_f_7863(f_10972, x_10975)
          in {res_10976},
        as_10973)
  let {i64 ret₂_10977} = n_10971
  in {res_10974}
}

fun {*[n_10978]i32} lifted_1_map1_7865 (i64 n_10978, i32 f_10979,
                                        bool map_10980,
                                        [n_10978]i64 as_10981) = {
  let {i32 lifted_1_map_arg_10982} =
    lifted_0_map_7861(map_10980, f_10979)
  let {[n_10978]i32 res_10983} =
    lifted_1_map_7864(n_10978, lifted_1_map_arg_10982, as_10981)
  in {res_10983}
}

fun {*[n_10985]i32} lifted_1_tabulate_7866 (bool map1_10984, i64 n_10985,
                                            i32 f_10986) = {
  let {[n_10985]i64 lifted_1_map1_arg_10987} =
    iota_3662(n_10985)
  let {i32 lifted_1_map1_arg_10988, bool lifted_1_map1_arg_10989} =
    lifted_0_map1_7860(map1_10984, f_10986)
  let {[n_10985]i32 res_10990} =
    lifted_1_map1_7865(n_10985, lifted_1_map1_arg_10988,
                       lifted_1_map1_arg_10989, lifted_1_map1_arg_10987)
  in {res_10990}
}

fun {[n_10991]i32} split_rng_7575 (i64 n_10991, i32 x_10992) = {
  let {i32 x_10993, i32 x_10994} =
    rand_7573(x_10992)
  let {i32 x_10995} = x_10993
  let {i32 nameless_10996} = x_10994
  let {bool lifted_1_tabulate_arg_10997, i64 lifted_1_tabulate_arg_10998} =
    tabulate_7707(n_10991)
  let {[lifted_1_tabulate_arg_10998]i32 res_10999} =
    lifted_1_tabulate_7866(lifted_1_tabulate_arg_10997,
                           lifted_1_tabulate_arg_10998, x_10995)
  let {bool dim_match_11000} = eq_i64(n_10991, lifted_1_tabulate_arg_10998)
  let {cert empty_or_match_cert_11001} =
    assert(dim_match_11000,
           "Function return value does not match shape of type [", n_10991,
           "]rng", "lib/github.com/diku-dk/cpprandom/random.fut:172:3-174:56")
  -- result_proper_shape_11002 aliases res_10999
  let {[n_10991]i32 result_proper_shape_11002} =
    <empty_or_match_cert_11001>
    reshape((~n_10991), res_10999)
  in {result_proper_shape_11002}
}

fun {i64} map_7709 (i64 d_11003) = {
  {d_11003}
}

fun {f32} -_3216 (f32 x_11004, f32 y_11005) = {
  let {f32 res_11006} = fsub32(x_11004, y_11005)
  in {res_11006}
}

fun {i64} u64_1914 (i64 x_11007) = {
  let {i64 x_11008} = zext i64 x_11007 to i64
  let {i64 res_11009} = zext i64 x_11008 to i64
  in {res_11009}
}

fun {f32} u64_3236 (i64 x_11010) = {
  let {i64 x_11011} =
    u64_1914(x_11010)
  let {f32 res_11012} = uitofp i64 x_11011 to f32
  in {res_11012}
}

fun {i64} unsign_2703 (i64 x_11013) = {
  let {i64 res_11014} = zext i64 x_11013 to i64
  in {res_11014}
}

fun {i64} i64_2786 (i64 x_11015) = {
  let {i64 unsign_arg_11016} = zext i64 x_11015 to i64
  let {i64 argdim₀_11017} = unsign_arg_11016
  let {i64 res_11018} =
    unsign_2703(unsign_arg_11016)
  in {res_11018}
}

fun {i64} to_i64_2562 (i32 x_11019) = {
  let {i32 x_11020} =
    sign_2489(x_11019)
  let {i64 res_11021} = zext i32 x_11020 to i64
  in {res_11021}
}

fun {f32} to_R_7671 (i32 x_11022) = {
  let {i64 i64_arg_11023} =
    to_i64_2562(x_11022)
  let {i64 u64_arg_11024} =
    i64_2786(i64_arg_11023)
  let {f32 res_11025} =
    u64_3236(u64_arg_11024)
  in {res_11025}
}

fun {f32} sqrt_3290 (f32 x_11026) = {
  let {f32 res_11027} =
    sqrt32(x_11026)
  in {res_11027}
}

fun {f32} *_3219 (f32 x_11028, f32 y_11029) = {
  let {f32 res_11030} = fmul32(x_11028, y_11029)
  in {res_11030}
}

fun {f32} i32_3242 (i32 x_11031) = {
  let {f32 res_11032} = sitofp i32 x_11031 to f32
  in {res_11032}
}

fun {f32} log_3292 (f32 x_11033) = {
  let {f32 res_11034} =
    log32(x_11033)
  in {res_11034}
}

fun {f32} +_3213 (f32 x_11035, f32 y_11036) = {
  let {f32 res_11037} = fadd32(x_11035, y_11036)
  in {res_11037}
}

fun {f32} cos_3302 (f32 x_11038) = {
  let {f32 res_11039} =
    cos32(x_11038)
  in {res_11039}
}

fun {i32, f32} rand_7676 (f32 mean_11040, f32 stddev_11041, i32 rng_11042) = {
  let {i32 res_11043, i32 res_11044} =
    rand_7573(rng_11042)
  let {i32 rng_11045} = res_11043
  let {i32 u1_11046} = res_11044
  let {i32 res_11047, i32 res_11048} =
    rand_7573(rng_11045)
  let {i32 rng_11049} = res_11047
  let {i32 u2_11050} = res_11048
  let {f32 -_arg_11051} =
    to_R_7671(res_10548)
  let {f32 -_arg_11052} =
    to_R_7671(2147483647i32)
  let {f32 /_arg_11053} =
    -_3216(-_arg_11052, -_arg_11051)
  let {f32 -_arg_11054} =
    to_R_7671(res_10548)
  let {f32 -_arg_11055} =
    to_R_7671(u1_11046)
  let {f32 /_arg_11056} =
    -_3216(-_arg_11055, -_arg_11054)
  let {f32 u1_11057} =
    /_3222(/_arg_11056, /_arg_11053)
  let {f32 u1_11058} = u1_11057
  let {f32 -_arg_11059} =
    to_R_7671(res_10548)
  let {f32 -_arg_11060} =
    to_R_7671(2147483647i32)
  let {f32 /_arg_11061} =
    -_3216(-_arg_11060, -_arg_11059)
  let {f32 -_arg_11062} =
    to_R_7671(res_10548)
  let {f32 -_arg_11063} =
    to_R_7671(u2_11050)
  let {f32 /_arg_11064} =
    -_3216(-_arg_11063, -_arg_11062)
  let {f32 u2_11065} =
    /_3222(/_arg_11064, /_arg_11061)
  let {f32 u2_11066} = u2_11065
  let {f32 *_arg_11067} =
    log_3292(u1_11058)
  let {i32 i32_arg_11068} = sub32(0i32, 2i32)
  let {f32 *_arg_11069} =
    i32_3242(i32_arg_11068)
  let {f32 sqrt_arg_11070} =
    *_3219(*_arg_11069, *_arg_11067)
  let {f32 r_11071} =
    sqrt_3290(sqrt_arg_11070)
  let {f32 r_11072} = r_11071
  let {f32 *_arg_11073} =
    i32_3242(2i32)
  let {f32 *_arg_11074} =
    *_3219(*_arg_11073, res_10549)
  let {f32 theta_11075} =
    *_3219(*_arg_11074, u2_11066)
  let {f32 theta_11076} = theta_11075
  let {f32 *_arg_11077} =
    cos_3302(theta_11076)
  let {f32 *_arg_11078} =
    *_3219(r_11072, *_arg_11077)
  let {f32 +_arg_11079} =
    *_3219(stddev_11041, *_arg_11078)
  let {f32 res_11080} =
    +_3213(mean_11040, +_arg_11079)
  in {rng_11049, res_11080}
}

fun {i64, i64} map_7713 (i64 d_11081, i64 d_11082) = {
  {d_11081, d_11082}
}

fun {f32} const_7719 (f32 x_11083, i64 nameless_11084) = {
  {x_11083}
}

fun {f32} const_7867 (f32 x_11085) = {
  {x_11085}
}

fun {f32} lifted_0_map_7868 (bool nameless_11086, f32 f_11087) = {
  {f_11087}
}

fun {f32} lifted_0_f_7870 (f32 x_11088, i64 nameless_11089) = {
  {x_11088}
}

fun {*[n_11090]f32} lifted_1_map_7871 (i64 n_11090, f32 f_11091,
                                       [n_11090]i64 as_11092) = {
  let {[n_11090]f32 res_11093} =
    map(n_11090,
        fn {f32} (i64 x_11094) =>
          let {f32 res_11095} =
            lifted_0_f_7870(f_11091, x_11094)
          in {res_11095},
        as_11092)
  let {i64 ret₂_11096} = n_11090
  in {res_11093}
}

fun {*[n_11097]f32} replicate_7720 (i64 n_11097, f32 x_11098) = {
  let {[n_11097]i64 lifted_1_map_arg_11099} =
    iota_3662(n_11097)
  let {f32 lifted_0_map_arg_11100} =
    const_7867(x_11098)
  let {f32 lifted_1_map_arg_11101} =
    lifted_0_map_7868(true, lifted_0_map_arg_11100)
  let {[n_11097]f32 res_11102} =
    lifted_1_map_7871(n_11097, lifted_1_map_arg_11101, lifted_1_map_arg_11099)
  in {res_11102}
}

fun {f32} shortstep_7265 (f32 a_11103, f32 b_11104, f32 deltat_11105,
                          f32 r0_11106, f32 sigma_11107, f32 r_11108,
                          f32 rand_11109) = {
  let {f32 y_11110} = fsub32(b_11104, r_11108)
  let {f32 x_11111} = fmul32(a_11103, y_11110)
  let {f32 x_11112} = fmul32(x_11111, deltat_11105)
  let {f32 x_11113} =
    sqrt_3290(deltat_11105)
  let {f32 x_11114} = fmul32(x_11113, rand_11109)
  let {f32 y_11115} = fmul32(x_11114, sigma_11107)
  let {f32 delta_r_11116} = fadd32(x_11112, y_11115)
  let {f32 delta_r_11117} = delta_r_11116
  let {f32 res_11118} = fadd32(r_11108, delta_r_11117)
  in {res_11118}
}

fun {[steps_11125]f32} mc_shortrate_7285 (f32 a_11119, f32 b_11120,
                                          f32 deltat_11121, f32 r0_11122,
                                          f32 sigma_11123, f32 r0_11124,
                                          i64 steps_11125,
                                          [steps_11125]f32 rands_11126) = {
  let {[steps_11125]f32 loop_init_11127} =
    replicate_7720(steps_11125, r0_11124)
  let {i64 upper_bound_11128} = sub64(steps_11125, 1i64)
  let {[steps_11125]f32 xs_11129} =
    -- Consumes loop_init_11127
    loop {*[steps_11125]f32 x_11131} = {loop_init_11127}
    for i_11130:i64 < upper_bound_11128 do {
      let {bool x_11132} = sle64(0i64, i_11130)
      let {bool y_11133} = slt64(i_11130, steps_11125)
      let {bool bounds_check_11134} = logand(x_11132, y_11133)
      let {cert index_certs_11135} =
        assert(bounds_check_11134, "Index [", i_11130,
                                   "] out of bounds for array of shape [",
                                   steps_11125, "].", "cva.fut:71:97-104")
      let {f32 shortstep_arg_11136} =
        <index_certs_11135>
        rands_11126[i_11130]
      let {bool x_11137} = sle64(0i64, i_11130)
      let {bool y_11138} = slt64(i_11130, steps_11125)
      let {bool bounds_check_11139} = logand(x_11137, y_11138)
      let {cert index_certs_11140} =
        assert(bounds_check_11139, "Index [", i_11130,
                                   "] out of bounds for array of shape [",
                                   steps_11125, "].", "cva.fut:71:92-95")
      let {f32 shortstep_arg_11141} =
        <index_certs_11140>
        x_11131[i_11130]
      let {f32 lw_val_11142} =
        shortstep_7265(a_11119, b_11120, deltat_11121, r0_11122, sigma_11123,
                       shortstep_arg_11141, shortstep_arg_11136)
      let {i64 i_11143} = add64(i_11130, 1i64)
      let {bool x_11144} = sle64(0i64, i_11143)
      let {bool y_11145} = slt64(i_11143, steps_11125)
      let {bool bounds_check_11146} = logand(x_11144, y_11145)
      let {cert index_certs_11147} =
        assert(bounds_check_11146, "Index [", i_11143,
                                   "] out of bounds for array of shape [",
                                   steps_11125, "].", "cva.fut:71:58-105")
      let {[steps_11125]f32 loopres_11148} =
        -- Consumes x_11131
        <index_certs_11147>
        x_11131 with [i_11143] <- lw_val_11142
      in {loopres_11148}
    }
  -- xs_11149 aliases xs_11129
  let {[steps_11125]f32 xs_11149} = xs_11129
  in {xs_11149}
}

fun {i64, i64, i64} map_7729 (i64 d_11150, i64 d_11151, i64 d_11152) = {
  {d_11150, d_11151, d_11152}
}

fun {i64} map_7732 (i64 d_11153) = {
  {d_11153}
}

fun {[n_11154]f32, [n_11154]f32} zip_7733 (i64 n_11154, [n_11154]f32 as_11155,
                                           [n_11154]f32 bs_11156) = {
  let {i64 ret₂_11157} = n_11154
  in {as_11155, bs_11156}
}

fun {[n_11158]f32, [n_11158]f32} zip2_7734 (i64 n_11158, [n_11158]f32 as_11159,
                                            [n_11158]f32 bs_11160) = {
  -- res_11161 aliases as_11159, bs_11160
  -- res_11162 aliases as_11159, bs_11160
  let {[n_11158]f32 res_11161, [n_11158]f32 res_11162} =
    zip_7733(n_11158, as_11159, bs_11160)
  in {res_11161, res_11162}
}

fun {i64} map2_7735 (i64 d_11163) = {
  {d_11163}
}

fun {[?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32} flatten_7741 (i64 n_11164, i64 m_11165,
                                              [n_11164][m_11165]f32 xs_11166,
                                              [n_11164][m_11165]f32 xs_11167,
                                              [n_11164][m_11165]f32 xs_11168,
                                              [n_11164][m_11165]i64 xs_11169,
                                              [n_11164][m_11165]f32 xs_11170,
                                              [n_11164][m_11165]f32 xs_11171,
                                              [n_11164][m_11165]f32 xs_11172,
                                              [n_11164][m_11165]f32 xs_11173,
                                              [n_11164][m_11165]f32 xs_11174,
                                              [n_11164][m_11165]f32 xs_11175,
                                              [n_11164][m_11165]f32 xs_11176) = {
  let {i64 flat_dim_11177} = mul_nw64(n_11164, m_11165)
  -- res_11178 aliases xs_11166
  let {[flat_dim_11177]f32 res_11178} = reshape((flat_dim_11177), xs_11166)
  let {i64 flat_dim_11179} = mul_nw64(n_11164, m_11165)
  -- res_11180 aliases xs_11167
  let {[flat_dim_11179]f32 res_11180} = reshape((flat_dim_11179), xs_11167)
  let {i64 flat_dim_11181} = mul_nw64(n_11164, m_11165)
  -- res_11182 aliases xs_11168
  let {[flat_dim_11181]f32 res_11182} = reshape((flat_dim_11181), xs_11168)
  let {i64 flat_dim_11183} = mul_nw64(n_11164, m_11165)
  -- res_11184 aliases xs_11169
  let {[flat_dim_11183]i64 res_11184} = reshape((flat_dim_11183), xs_11169)
  let {i64 flat_dim_11185} = mul_nw64(n_11164, m_11165)
  -- res_11186 aliases xs_11170
  let {[flat_dim_11185]f32 res_11186} = reshape((flat_dim_11185), xs_11170)
  let {i64 flat_dim_11187} = mul_nw64(n_11164, m_11165)
  -- res_11188 aliases xs_11171
  let {[flat_dim_11187]f32 res_11188} = reshape((flat_dim_11187), xs_11171)
  let {i64 flat_dim_11189} = mul_nw64(n_11164, m_11165)
  -- res_11190 aliases xs_11172
  let {[flat_dim_11189]f32 res_11190} = reshape((flat_dim_11189), xs_11172)
  let {i64 flat_dim_11191} = mul_nw64(n_11164, m_11165)
  -- res_11192 aliases xs_11173
  let {[flat_dim_11191]f32 res_11192} = reshape((flat_dim_11191), xs_11173)
  let {i64 flat_dim_11193} = mul_nw64(n_11164, m_11165)
  -- res_11194 aliases xs_11174
  let {[flat_dim_11193]f32 res_11194} = reshape((flat_dim_11193), xs_11174)
  let {i64 flat_dim_11195} = mul_nw64(n_11164, m_11165)
  -- res_11196 aliases xs_11175
  let {[flat_dim_11195]f32 res_11196} = reshape((flat_dim_11195), xs_11175)
  let {i64 flat_dim_11197} = mul_nw64(n_11164, m_11165)
  -- res_11198 aliases xs_11176
  let {[flat_dim_11197]f32 res_11198} = reshape((flat_dim_11197), xs_11176)
  let {i64 ret₂_11199} = flat_dim_11177
  let {bool dim_match_11200} = eq_i64(flat_dim_11177, flat_dim_11179)
  let {cert empty_or_match_cert_11201} =
    assert(dim_match_11200,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11202 aliases res_11180
  let {[flat_dim_11177]f32 result_proper_shape_11202} =
    <empty_or_match_cert_11201>
    reshape((~flat_dim_11177), res_11180)
  let {bool dim_match_11203} = eq_i64(flat_dim_11177, flat_dim_11181)
  let {cert empty_or_match_cert_11204} =
    assert(dim_match_11203,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11205 aliases res_11182
  let {[flat_dim_11177]f32 result_proper_shape_11205} =
    <empty_or_match_cert_11204>
    reshape((~flat_dim_11177), res_11182)
  let {bool dim_match_11206} = eq_i64(flat_dim_11177, flat_dim_11183)
  let {cert empty_or_match_cert_11207} =
    assert(dim_match_11206,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11208 aliases res_11184
  let {[flat_dim_11177]i64 result_proper_shape_11208} =
    <empty_or_match_cert_11207>
    reshape((~flat_dim_11177), res_11184)
  let {bool dim_match_11209} = eq_i64(flat_dim_11177, flat_dim_11185)
  let {cert empty_or_match_cert_11210} =
    assert(dim_match_11209,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11211 aliases res_11186
  let {[flat_dim_11177]f32 result_proper_shape_11211} =
    <empty_or_match_cert_11210>
    reshape((~flat_dim_11177), res_11186)
  let {bool dim_match_11212} = eq_i64(flat_dim_11177, flat_dim_11187)
  let {cert empty_or_match_cert_11213} =
    assert(dim_match_11212,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11214 aliases res_11188
  let {[flat_dim_11177]f32 result_proper_shape_11214} =
    <empty_or_match_cert_11213>
    reshape((~flat_dim_11177), res_11188)
  let {bool dim_match_11215} = eq_i64(flat_dim_11177, flat_dim_11189)
  let {cert empty_or_match_cert_11216} =
    assert(dim_match_11215,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11217 aliases res_11190
  let {[flat_dim_11177]f32 result_proper_shape_11217} =
    <empty_or_match_cert_11216>
    reshape((~flat_dim_11177), res_11190)
  let {bool dim_match_11218} = eq_i64(flat_dim_11177, flat_dim_11191)
  let {cert empty_or_match_cert_11219} =
    assert(dim_match_11218,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11220 aliases res_11192
  let {[flat_dim_11177]f32 result_proper_shape_11220} =
    <empty_or_match_cert_11219>
    reshape((~flat_dim_11177), res_11192)
  let {bool dim_match_11221} = eq_i64(flat_dim_11177, flat_dim_11193)
  let {cert empty_or_match_cert_11222} =
    assert(dim_match_11221,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11223 aliases res_11194
  let {[flat_dim_11177]f32 result_proper_shape_11223} =
    <empty_or_match_cert_11222>
    reshape((~flat_dim_11177), res_11194)
  let {bool dim_match_11224} = eq_i64(flat_dim_11177, flat_dim_11195)
  let {cert empty_or_match_cert_11225} =
    assert(dim_match_11224,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11226 aliases res_11196
  let {[flat_dim_11177]f32 result_proper_shape_11226} =
    <empty_or_match_cert_11225>
    reshape((~flat_dim_11177), res_11196)
  let {bool dim_match_11227} = eq_i64(flat_dim_11177, flat_dim_11197)
  let {cert empty_or_match_cert_11228} =
    assert(dim_match_11227,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11229 aliases res_11198
  let {[flat_dim_11177]f32 result_proper_shape_11229} =
    <empty_or_match_cert_11228>
    reshape((~flat_dim_11177), res_11198)
  in {flat_dim_11177, res_11178, result_proper_shape_11202,
      result_proper_shape_11205, result_proper_shape_11208,
      result_proper_shape_11211, result_proper_shape_11214,
      result_proper_shape_11217, result_proper_shape_11220,
      result_proper_shape_11223, result_proper_shape_11226,
      result_proper_shape_11229}
}

fun {[?0][d_11232]f32, [?0][d_11232]f32, [?0][d_11232]f32, [?0][d_11232]i64,
     [?0][d_11232]f32, [?0][d_11232]f32, [?0][d_11232]f32, [?0][d_11232]f32,
     [?0][d_11232]f32, [?0][d_11232]f32, [?0][d_11232]f32}
flatten_7743 (i64 n_11230, i64 m_11231, i64 d_11232,
              [n_11230][m_11231][d_11232]f32 xs_11233,
              [n_11230][m_11231][d_11232]f32 xs_11234,
              [n_11230][m_11231][d_11232]f32 xs_11235,
              [n_11230][m_11231][d_11232]i64 xs_11236,
              [n_11230][m_11231][d_11232]f32 xs_11237,
              [n_11230][m_11231][d_11232]f32 xs_11238,
              [n_11230][m_11231][d_11232]f32 xs_11239,
              [n_11230][m_11231][d_11232]f32 xs_11240,
              [n_11230][m_11231][d_11232]f32 xs_11241,
              [n_11230][m_11231][d_11232]f32 xs_11242,
              [n_11230][m_11231][d_11232]f32 xs_11243) = {
  let {i64 flat_dim_11244} = mul_nw64(n_11230, m_11231)
  -- res_11245 aliases xs_11233
  let {[flat_dim_11244][d_11232]f32 res_11245} = reshape((flat_dim_11244,
                                                          d_11232), xs_11233)
  let {i64 flat_dim_11246} = mul_nw64(n_11230, m_11231)
  -- res_11247 aliases xs_11234
  let {[flat_dim_11246][d_11232]f32 res_11247} = reshape((flat_dim_11246,
                                                          d_11232), xs_11234)
  let {i64 flat_dim_11248} = mul_nw64(n_11230, m_11231)
  -- res_11249 aliases xs_11235
  let {[flat_dim_11248][d_11232]f32 res_11249} = reshape((flat_dim_11248,
                                                          d_11232), xs_11235)
  let {i64 flat_dim_11250} = mul_nw64(n_11230, m_11231)
  -- res_11251 aliases xs_11236
  let {[flat_dim_11250][d_11232]i64 res_11251} = reshape((flat_dim_11250,
                                                          d_11232), xs_11236)
  let {i64 flat_dim_11252} = mul_nw64(n_11230, m_11231)
  -- res_11253 aliases xs_11237
  let {[flat_dim_11252][d_11232]f32 res_11253} = reshape((flat_dim_11252,
                                                          d_11232), xs_11237)
  let {i64 flat_dim_11254} = mul_nw64(n_11230, m_11231)
  -- res_11255 aliases xs_11238
  let {[flat_dim_11254][d_11232]f32 res_11255} = reshape((flat_dim_11254,
                                                          d_11232), xs_11238)
  let {i64 flat_dim_11256} = mul_nw64(n_11230, m_11231)
  -- res_11257 aliases xs_11239
  let {[flat_dim_11256][d_11232]f32 res_11257} = reshape((flat_dim_11256,
                                                          d_11232), xs_11239)
  let {i64 flat_dim_11258} = mul_nw64(n_11230, m_11231)
  -- res_11259 aliases xs_11240
  let {[flat_dim_11258][d_11232]f32 res_11259} = reshape((flat_dim_11258,
                                                          d_11232), xs_11240)
  let {i64 flat_dim_11260} = mul_nw64(n_11230, m_11231)
  -- res_11261 aliases xs_11241
  let {[flat_dim_11260][d_11232]f32 res_11261} = reshape((flat_dim_11260,
                                                          d_11232), xs_11241)
  let {i64 flat_dim_11262} = mul_nw64(n_11230, m_11231)
  -- res_11263 aliases xs_11242
  let {[flat_dim_11262][d_11232]f32 res_11263} = reshape((flat_dim_11262,
                                                          d_11232), xs_11242)
  let {i64 flat_dim_11264} = mul_nw64(n_11230, m_11231)
  -- res_11265 aliases xs_11243
  let {[flat_dim_11264][d_11232]f32 res_11265} = reshape((flat_dim_11264,
                                                          d_11232), xs_11243)
  let {i64 ret₂_11266} = flat_dim_11244
  let {bool dim_match_11267} = eq_i64(flat_dim_11244, flat_dim_11246)
  let {bool dim_match_11268} = eq_i64(d_11232, d_11232)
  let {bool match_11269} = logand(dim_match_11268, dim_match_11267)
  let {cert empty_or_match_cert_11270} =
    assert(match_11269,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11271 aliases res_11247
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11271} =
    <empty_or_match_cert_11270>
    reshape((~flat_dim_11244, ~d_11232), res_11247)
  let {bool dim_match_11272} = eq_i64(flat_dim_11244, flat_dim_11248)
  let {bool dim_match_11273} = eq_i64(d_11232, d_11232)
  let {bool match_11274} = logand(dim_match_11273, dim_match_11272)
  let {cert empty_or_match_cert_11275} =
    assert(match_11274,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11276 aliases res_11249
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11276} =
    <empty_or_match_cert_11275>
    reshape((~flat_dim_11244, ~d_11232), res_11249)
  let {bool dim_match_11277} = eq_i64(flat_dim_11244, flat_dim_11250)
  let {bool dim_match_11278} = eq_i64(d_11232, d_11232)
  let {bool match_11279} = logand(dim_match_11278, dim_match_11277)
  let {cert empty_or_match_cert_11280} =
    assert(match_11279,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11281 aliases res_11251
  let {[flat_dim_11244][d_11232]i64 result_proper_shape_11281} =
    <empty_or_match_cert_11280>
    reshape((~flat_dim_11244, ~d_11232), res_11251)
  let {bool dim_match_11282} = eq_i64(flat_dim_11244, flat_dim_11252)
  let {bool dim_match_11283} = eq_i64(d_11232, d_11232)
  let {bool match_11284} = logand(dim_match_11283, dim_match_11282)
  let {cert empty_or_match_cert_11285} =
    assert(match_11284,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11286 aliases res_11253
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11286} =
    <empty_or_match_cert_11285>
    reshape((~flat_dim_11244, ~d_11232), res_11253)
  let {bool dim_match_11287} = eq_i64(flat_dim_11244, flat_dim_11254)
  let {bool dim_match_11288} = eq_i64(d_11232, d_11232)
  let {bool match_11289} = logand(dim_match_11288, dim_match_11287)
  let {cert empty_or_match_cert_11290} =
    assert(match_11289,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11291 aliases res_11255
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11291} =
    <empty_or_match_cert_11290>
    reshape((~flat_dim_11244, ~d_11232), res_11255)
  let {bool dim_match_11292} = eq_i64(flat_dim_11244, flat_dim_11256)
  let {bool dim_match_11293} = eq_i64(d_11232, d_11232)
  let {bool match_11294} = logand(dim_match_11293, dim_match_11292)
  let {cert empty_or_match_cert_11295} =
    assert(match_11294,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11296 aliases res_11257
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11296} =
    <empty_or_match_cert_11295>
    reshape((~flat_dim_11244, ~d_11232), res_11257)
  let {bool dim_match_11297} = eq_i64(flat_dim_11244, flat_dim_11258)
  let {bool dim_match_11298} = eq_i64(d_11232, d_11232)
  let {bool match_11299} = logand(dim_match_11298, dim_match_11297)
  let {cert empty_or_match_cert_11300} =
    assert(match_11299,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11301 aliases res_11259
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11301} =
    <empty_or_match_cert_11300>
    reshape((~flat_dim_11244, ~d_11232), res_11259)
  let {bool dim_match_11302} = eq_i64(flat_dim_11244, flat_dim_11260)
  let {bool dim_match_11303} = eq_i64(d_11232, d_11232)
  let {bool match_11304} = logand(dim_match_11303, dim_match_11302)
  let {cert empty_or_match_cert_11305} =
    assert(match_11304,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11306 aliases res_11261
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11306} =
    <empty_or_match_cert_11305>
    reshape((~flat_dim_11244, ~d_11232), res_11261)
  let {bool dim_match_11307} = eq_i64(flat_dim_11244, flat_dim_11262)
  let {bool dim_match_11308} = eq_i64(d_11232, d_11232)
  let {bool match_11309} = logand(dim_match_11308, dim_match_11307)
  let {cert empty_or_match_cert_11310} =
    assert(match_11309,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11311 aliases res_11263
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11311} =
    <empty_or_match_cert_11310>
    reshape((~flat_dim_11244, ~d_11232), res_11263)
  let {bool dim_match_11312} = eq_i64(flat_dim_11244, flat_dim_11264)
  let {bool dim_match_11313} = eq_i64(d_11232, d_11232)
  let {bool match_11314} = logand(dim_match_11313, dim_match_11312)
  let {cert empty_or_match_cert_11315} =
    assert(match_11314,
           "Function return value does not match shape of type []t",
           "/prelude/array.fut:77:1-78:23")
  -- result_proper_shape_11316 aliases res_11265
  let {[flat_dim_11244][d_11232]f32 result_proper_shape_11316} =
    <empty_or_match_cert_11315>
    reshape((~flat_dim_11244, ~d_11232), res_11265)
  in {flat_dim_11244, res_11245, result_proper_shape_11271,
      result_proper_shape_11276, result_proper_shape_11281,
      result_proper_shape_11286, result_proper_shape_11291,
      result_proper_shape_11296, result_proper_shape_11301,
      result_proper_shape_11306, result_proper_shape_11311,
      result_proper_shape_11316}
}

fun {[?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32} flatten_3d_7744 (i64 n_11317, i64 m_11318,
                                                 i64 l_11319,
                                                 [n_11317][m_11318][l_11319]f32 xs_11320,
                                                 [n_11317][m_11318][l_11319]f32 xs_11321,
                                                 [n_11317][m_11318][l_11319]f32 xs_11322,
                                                 [n_11317][m_11318][l_11319]i64 xs_11323,
                                                 [n_11317][m_11318][l_11319]f32 xs_11324,
                                                 [n_11317][m_11318][l_11319]f32 xs_11325,
                                                 [n_11317][m_11318][l_11319]f32 xs_11326,
                                                 [n_11317][m_11318][l_11319]f32 xs_11327,
                                                 [n_11317][m_11318][l_11319]f32 xs_11328,
                                                 [n_11317][m_11318][l_11319]f32 xs_11329,
                                                 [n_11317][m_11318][l_11319]f32 xs_11330) = {
  -- flatten_arg_11332 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11333 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11334 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11335 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11336 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11337 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11338 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11339 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11340 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11341 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  -- flatten_arg_11342 aliases xs_11320, xs_11321, xs_11322, xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328, xs_11329, xs_11330
  let {i64 size_11331;
       [size_11331][l_11319]f32 flatten_arg_11332,
       [size_11331][l_11319]f32 flatten_arg_11333,
       [size_11331][l_11319]f32 flatten_arg_11334,
       [size_11331][l_11319]i64 flatten_arg_11335,
       [size_11331][l_11319]f32 flatten_arg_11336,
       [size_11331][l_11319]f32 flatten_arg_11337,
       [size_11331][l_11319]f32 flatten_arg_11338,
       [size_11331][l_11319]f32 flatten_arg_11339,
       [size_11331][l_11319]f32 flatten_arg_11340,
       [size_11331][l_11319]f32 flatten_arg_11341,
       [size_11331][l_11319]f32 flatten_arg_11342} =
    flatten_7743(n_11317, m_11318, l_11319, xs_11320, xs_11321, xs_11322,
                 xs_11323, xs_11324, xs_11325, xs_11326, xs_11327, xs_11328,
                 xs_11329, xs_11330)
  let {i64 ret₇_11343} = size_11331
  -- res_11345 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11346 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11347 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11348 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11349 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11350 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11351 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11352 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11353 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11354 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  -- res_11355 aliases flatten_arg_11332, flatten_arg_11333, flatten_arg_11334, flatten_arg_11335, flatten_arg_11336, flatten_arg_11337, flatten_arg_11338, flatten_arg_11339, flatten_arg_11340, flatten_arg_11341, flatten_arg_11342
  let {i64 size_11344;
       [size_11344]f32 res_11345, [size_11344]f32 res_11346,
       [size_11344]f32 res_11347, [size_11344]i64 res_11348,
       [size_11344]f32 res_11349, [size_11344]f32 res_11350,
       [size_11344]f32 res_11351, [size_11344]f32 res_11352,
       [size_11344]f32 res_11353, [size_11344]f32 res_11354,
       [size_11344]f32 res_11355} =
    flatten_7741(size_11331, l_11319, flatten_arg_11332, flatten_arg_11333,
                 flatten_arg_11334, flatten_arg_11335, flatten_arg_11336,
                 flatten_arg_11337, flatten_arg_11338, flatten_arg_11339,
                 flatten_arg_11340, flatten_arg_11341, flatten_arg_11342)
  let {i64 ret₈_11356} = size_11344
  in {size_11344, res_11345, res_11346, res_11347, res_11348, res_11349,
      res_11350, res_11351, res_11352, res_11353, res_11354, res_11355}
}

fun {[n_11357]i64, [n_11357]i64} zip_7748 (i64 n_11357, [n_11357]i64 as_11358,
                                           [n_11357]i64 bs_11359) = {
  let {i64 ret₂_11360} = n_11357
  in {as_11358, bs_11359}
}

fun {[n_11361]i64, [n_11361]i64} zip2_7749 (i64 n_11361, [n_11361]i64 as_11362,
                                            [n_11361]i64 bs_11363) = {
  -- res_11364 aliases as_11362, bs_11363
  -- res_11365 aliases as_11362, bs_11363
  let {[n_11361]i64 res_11364, [n_11361]i64 res_11365} =
    zip_7748(n_11361, as_11362, bs_11363)
  in {res_11364, res_11365}
}

fun {[n_11366]i64} rotate_7751 (i64 n_11366, i64 r_11367,
                                [n_11366]i64 xs_11368) = {
  -- res_11369 aliases xs_11368
  let {[n_11366]i64 res_11369} = rotate((r_11367), xs_11368)
  let {i64 ret₁_11370} = n_11366
  in {res_11369}
}

fun {*[?0]i64} reduce_by_index_7752 (i64 m_11371, *[m_11371]i64 dest_11372) = {
  {m_11371, dest_11372}
}

fun {i64} const_7754 (i64 x_11373, i64 nameless_11374) = {
  {x_11373}
}

fun {i64} const_7872 (i64 x_11375) = {
  {x_11375}
}

fun {i64} lifted_0_map_7873 (bool nameless_11376, i64 f_11377) = {
  {f_11377}
}

fun {i64} lifted_0_f_7875 (i64 x_11378, i64 nameless_11379) = {
  {x_11378}
}

fun {*[n_11380]i64} lifted_1_map_7876 (i64 n_11380, i64 f_11381,
                                       [n_11380]i64 as_11382) = {
  let {[n_11380]i64 res_11383} =
    map(n_11380,
        fn {i64} (i64 x_11384) =>
          let {i64 res_11385} =
            lifted_0_f_7875(f_11381, x_11384)
          in {res_11385},
        as_11382)
  let {i64 ret₂_11386} = n_11380
  in {res_11383}
}

fun {*[n_11387]i64} replicate_7755 (i64 n_11387, i64 x_11388) = {
  let {[n_11387]i64 lifted_1_map_arg_11389} =
    iota_3662(n_11387)
  let {i64 lifted_0_map_arg_11390} =
    const_7872(x_11388)
  let {i64 lifted_1_map_arg_11391} =
    lifted_0_map_7873(true, lifted_0_map_arg_11390)
  let {[n_11387]i64 res_11392} =
    lifted_1_map_7876(n_11387, lifted_1_map_arg_11391, lifted_1_map_arg_11389)
  in {res_11392}
}

fun {i64} max_1951 (i64 x_11393, i64 y_11394) = {
  let {i64 res_11395} = smax64(x_11393, y_11394)
  in {res_11395}
}

fun {[n_11396]bool, [n_11396]i64} unzip_7759 (i64 n_11396,
                                              [n_11396]bool xs_11397,
                                              [n_11396]i64 xs_11398) = {
  let {i64 ret₂_11399} = n_11396
  let {i64 ret₃_11400} = n_11396
  in {xs_11397, xs_11398}
}

fun {[n_11401]bool, [n_11401]i64} zip_7763 (i64 n_11401, [n_11401]bool as_11402,
                                            [n_11401]i64 bs_11403) = {
  let {i64 ret₂_11404} = n_11401
  in {as_11402, bs_11403}
}

fun {bool} lifted_0_scan_7879 (bool nameless_11405, bool op_11406) = {
  {op_11406}
}

fun {i64, bool} lifted_1_scan_7880 (bool op_11407, i64 ne_11408) = {
  {ne_11408, op_11407}
}

fun {i64} lifted_0_op_7883 (bool nameless_11409, i64 x_11410) = {
  {x_11410}
}

fun {i64} lifted_1_op_7884 (i64 x_11411, i64 x_11412) = {
  let {i64 res_11413} = add64(x_11411, x_11412)
  in {res_11413}
}

fun {*[n_11414]i64} lifted_2_scan_7885 (i64 n_11414, i64 ne_11415,
                                        bool op_11416,
                                        [n_11414]i64 as_11417) = {
  let {[n_11414]i64 res_11418} =
    scanomap(n_11414,
             {fn {i64} (i64 x_11419, i64 x_11420) =>
                let {i64 lifted_1_op_arg_11421} =
                  lifted_0_op_7883(op_11416, x_11419)
                let {i64 res_11422} =
                  lifted_1_op_7884(lifted_1_op_arg_11421, x_11420)
                in {res_11422},
              {ne_11415}},
             fn {i64} (i64 x_11423) =>
               {x_11423},
             as_11417)
  let {i64 ret₁_11424} = n_11414
  in {res_11418}
}

fun {bool, bool} lifted_0_map2_7886 (bool map_11425, bool f_11426) = {
  {f_11426, map_11425}
}

fun {[n_11427]i64, bool, bool} lifted_1_map2_7887 (i64 n_11427, bool f_11428,
                                                   bool map_11429,
                                                   [n_11427]i64 as_11430) = {
  {as_11430, f_11428, map_11429}
}

fun {bool} lifted_0_map_7888 (bool nameless_11431, bool f_11432) = {
  {f_11432}
}

fun {i64} lifted_0_f_7890 (bool nameless_11433, i64 i_11434) = {
  {i_11434}
}

fun {i64} lifted_1_f_7891 (i64 i_11435, i64 x_11436) = {
  let {bool cond_11437} = eq_i64(i_11435, 0i64)
  let {i64 res_11438} =
    -- Branch returns: {i64}
    if cond_11437
    then {0i64} else {x_11436}
  in {res_11438}
}

fun {i64} lifted_0_f_7892 (bool f_11439, i64 a_11440, i64 b_11441) = {
  let {i64 lifted_1_f_arg_11442} =
    lifted_0_f_7890(f_11439, a_11440)
  let {i64 res_11443} =
    lifted_1_f_7891(lifted_1_f_arg_11442, b_11441)
  in {res_11443}
}

fun {*[n_11444]i64} lifted_1_map_7893 (i64 n_11444, bool f_11445,
                                       [n_11444]i64 as_11446,
                                       [n_11444]i64 as_11447) = {
  let {[n_11444]i64 res_11448} =
    map(n_11444,
        fn {i64} (i64 x_11449, i64 x_11450) =>
          let {i64 res_11451} =
            lifted_0_f_7892(f_11445, x_11449, x_11450)
          in {res_11451},
        as_11446, as_11447)
  let {i64 ret₂_11452} = n_11444
  in {res_11448}
}

fun {*[n_11453]i64} lifted_2_map2_7894 (i64 n_11453, [n_11453]i64 as_11454,
                                        bool f_11455, bool map_11456,
                                        [n_11453]i64 bs_11457) = {
  -- lifted_1_map_arg_11458 aliases as_11454, bs_11457
  -- lifted_1_map_arg_11459 aliases as_11454, bs_11457
  let {[n_11453]i64 lifted_1_map_arg_11458,
       [n_11453]i64 lifted_1_map_arg_11459} =
    zip2_7749(n_11453, as_11454, bs_11457)
  let {bool lifted_1_map_arg_11460} =
    lifted_0_map_7888(map_11456, f_11455)
  let {[n_11453]i64 res_11461} =
    lifted_1_map_7893(n_11453, lifted_1_map_arg_11460, lifted_1_map_arg_11458,
                      lifted_1_map_arg_11459)
  in {res_11461}
}

fun {bool} lifted_0_reduce_7897 (bool nameless_11462, bool op_11463) = {
  {op_11463}
}

fun {i64, bool} lifted_1_reduce_7898 (bool op_11464, i64 ne_11465) = {
  {ne_11465, op_11464}
}

fun {i64} lifted_0_op_7901 (bool nameless_11466, i64 x_11467) = {
  {x_11467}
}

fun {i64} lifted_1_op_7902 (i64 x_11468, i64 x_11469) = {
  let {i64 res_11470} = add64(x_11468, x_11469)
  in {res_11470}
}

fun {i64} lifted_2_reduce_7903 (i64 n_11471, i64 ne_11472, bool op_11473,
                                [n_11471]i64 as_11474) = {
  let {i64 res_11475} =
    redomap(n_11471,
            {fn {i64} (i64 x_11476, i64 x_11477) =>
               let {i64 lifted_1_op_arg_11478} =
                 lifted_0_op_7901(op_11473, x_11476)
               let {i64 res_11479} =
                 lifted_1_op_7902(lifted_1_op_arg_11478, x_11477)
               in {res_11479},
             {ne_11472}},
            fn {i64} (i64 x_11480) =>
              {x_11480},
            as_11474)
  in {res_11475}
}

fun {*[m_11481]i64, bool} lifted_1_reduce_by_index_7904 (i64 m_11481,
                                                         *[m_11481]i64 dest_11482,
                                                         bool f_11483) = {
  {dest_11482, f_11483}
}

fun {*[m_11484]i64, bool, i64} lifted_2_reduce_by_index_7905 (i64 m_11484,
                                                              *[m_11484]i64 dest_11485,
                                                              bool f_11486,
                                                              i64 ne_11487) = {
  {dest_11485, f_11486, ne_11487}
}

fun {*[m_11489]i64, bool, [n_11488]i64, i64}
lifted_3_reduce_by_index_7906 (i64 n_11488, i64 m_11489,
                               *[m_11489]i64 dest_11490, bool f_11491,
                               i64 ne_11492, [n_11488]i64 is_11493) = {
  {dest_11490, f_11491, is_11493, ne_11492}
}

fun {i64} lifted_0_f_7909 (bool nameless_11494, i64 x_11495) = {
  {x_11495}
}

fun {i64} lifted_1_f_7910 (i64 x_11496, i64 y_11497) = {
  let {i64 res_11498} = smax64(x_11496, y_11497)
  in {res_11498}
}

fun {*[m_11499]i64} lifted_4_reduce_by_index_7911 (i64 m_11499, i64 n_11500,
                                                   *[m_11499]i64 dest_11501,
                                                   bool f_11502,
                                                   [n_11500]i64 is_11503,
                                                   i64 ne_11504,
                                                   [n_11500]i64 as_11505) = {
  let {bool bucket_cmp_11506} = eq_i64(n_11500, n_11500)
  let {cert bucket_cert_11507} =
    assert(bucket_cmp_11506, "length of index and value array does not match",
           "/prelude/soacs.fut:122:3-42")
  -- is_11508 aliases is_11503
  let {[n_11500]i64 is_11508} =
    <bucket_cert_11507>
    reshape((~n_11500), is_11503)
  let {[m_11499]i64 res_11509} =
    -- Consumes dest_11501
    hist(n_11500,
         {m_11499, 1i64, {dest_11501},
          {ne_11504},
          fn {i64} (i64 x_11510, i64 x_11511) =>
            let {i64 lifted_1_f_arg_11512} =
              lifted_0_f_7909(f_11502, x_11510)
            let {i64 res_11513} =
              lifted_1_f_7910(lifted_1_f_arg_11512, x_11511)
            in {res_11513}},
         fn {i64, i64} (i64 bucket_p_11514, i64 img_p_11515) =>
           {bucket_p_11514, img_p_11515},
         is_11508, as_11505)
  let {i64 ret₂_11516} = m_11499
  in {res_11509}
}

fun {bool} lifted_0_map_7912 (bool nameless_11517, bool f_11518) = {
  {f_11518}
}

fun {bool} lifted_0_f_7914 (bool nameless_11519, i64 x_11520) = {
  let {bool res_11521} = slt64(0i64, x_11520)
  in {res_11521}
}

fun {*[n_11522]bool} lifted_1_map_7915 (i64 n_11522, bool f_11523,
                                        [n_11522]i64 as_11524) = {
  let {[n_11522]bool res_11525} =
    map(n_11522,
        fn {bool} (i64 x_11526) =>
          let {bool res_11527} =
            lifted_0_f_7914(f_11523, x_11526)
          in {res_11527},
        as_11524)
  let {i64 ret₂_11528} = n_11522
  in {res_11525}
}

fun {bool, bool} lifted_0_segmented_scan_7918 (bool scan_11529,
                                               bool op_11530) = {
  {op_11530, scan_11529}
}

fun {i64, bool, bool} lifted_1_segmented_scan_7919 (bool op_11531,
                                                    bool scan_11532,
                                                    i64 ne_11533) = {
  {ne_11533, op_11531, scan_11532}
}

fun {[n_11534]bool, i64, bool, bool} lifted_2_segmented_scan_7920 (i64 n_11534,
                                                                   i64 ne_11535,
                                                                   bool op_11536,
                                                                   bool scan_11537,
                                                                   [n_11534]bool flags_11538) = {
  {flags_11538, ne_11535, op_11536, scan_11537}
}

fun {bool} lifted_0_scan_7921 (bool nameless_11539, bool op_11540) = {
  {op_11540}
}

fun {bool, i64, bool} lifted_1_scan_7922 (bool op_11541, bool 0_11542,
                                          i64 1_11543) = {
  {0_11542, 1_11543, op_11541}
}

fun {bool, i64, bool} lifted_0_op_7925 (bool op_11544, bool x_flag_11545,
                                        i64 x_11546) = {
  {op_11544, x_11546, x_flag_11545}
}

fun {i64} lifted_0_op_7926 (bool nameless_11547, i64 x_11548) = {
  {x_11548}
}

fun {i64} lifted_1_op_7927 (i64 x_11549, i64 x_11550) = {
  let {i64 res_11551} = add64(x_11549, x_11550)
  in {res_11551}
}

fun {bool, i64} lifted_1_op_7928 (bool op_11552, i64 x_11553, bool x_flag_11554,
                                  bool y_flag_11555, i64 y_11556) = {
  let {bool res_11557} =
    -- Branch returns: {bool}
    if x_flag_11554
    then {true} else {y_flag_11555}
  let {i64 res_11558} =
    -- Branch returns: {i64}
    if y_flag_11555
    then {y_11556} else {
      let {i64 lifted_1_op_arg_11559} =
        lifted_0_op_7926(op_11552, x_11553)
      let {i64 res_11560} =
        lifted_1_op_7927(lifted_1_op_arg_11559, y_11556)
      in {res_11560}
    }
  in {res_11557, res_11558}
}

fun {*[n_11561]bool, *[n_11561]i64} lifted_2_scan_7929 (i64 n_11561,
                                                        bool 0_11562,
                                                        i64 1_11563,
                                                        bool op_11564,
                                                        [n_11561]bool as_11565,
                                                        [n_11561]i64 as_11566) = {
  let {[n_11561]bool res_11567, [n_11561]i64 res_11568} =
    scanomap(n_11561,
             {fn {bool, i64} (bool x_11569, i64 x_11570, bool x_11571,
                              i64 x_11572) =>
                let {bool lifted_1_op_arg_11573, i64 lifted_1_op_arg_11574,
                     bool lifted_1_op_arg_11575} =
                  lifted_0_op_7925(op_11564, x_11569, x_11570)
                let {bool res_11576, i64 res_11577} =
                  lifted_1_op_7928(lifted_1_op_arg_11573, lifted_1_op_arg_11574,
                                   lifted_1_op_arg_11575, x_11571, x_11572)
                in {res_11576, res_11577},
              {0_11562, 1_11563}},
             fn {bool, i64} (bool x_11578, i64 x_11579) =>
               {x_11578, x_11579},
             as_11565, as_11566)
  let {i64 ret₁_11580} = n_11561
  in {res_11567, res_11568}
}

fun {[n_11581]i64} lifted_3_segmented_scan_7930 (i64 n_11581,
                                                 [n_11581]bool flags_11582,
                                                 i64 ne_11583, bool op_11584,
                                                 bool scan_11585,
                                                 [n_11581]i64 as_11586) = {
  -- lifted_2_scan_arg_11587 aliases flags_11582, as_11586
  -- lifted_2_scan_arg_11588 aliases flags_11582, as_11586
  let {[n_11581]bool lifted_2_scan_arg_11587,
       [n_11581]i64 lifted_2_scan_arg_11588} =
    zip_7763(n_11581, flags_11582, as_11586)
  let {bool lifted_1_scan_arg_11589} =
    lifted_0_scan_7921(scan_11585, op_11584)
  let {bool lifted_2_scan_arg_11590, i64 lifted_2_scan_arg_11591,
       bool lifted_2_scan_arg_11592} =
    lifted_1_scan_7922(lifted_1_scan_arg_11589, false, ne_11583)
  let {[n_11581]bool unzip_arg_11593, [n_11581]i64 unzip_arg_11594} =
    lifted_2_scan_7929(n_11581, lifted_2_scan_arg_11590,
                       lifted_2_scan_arg_11591, lifted_2_scan_arg_11592,
                       lifted_2_scan_arg_11587, lifted_2_scan_arg_11588)
  -- res_11595 aliases unzip_arg_11593, unzip_arg_11594
  -- res_11596 aliases unzip_arg_11593, unzip_arg_11594
  let {[n_11581]bool res_11595, [n_11581]i64 res_11596} =
    unzip_7759(n_11581, unzip_arg_11593, unzip_arg_11594)
  in {res_11596}
}

fun {[?0]i64} replicated_iota_7765 (i64 n_11597, [n_11597]i64 reps_11598) = {
  let {bool lifted_1_scan_arg_11599} =
    lifted_0_scan_7879(true, true)
  let {i64 lifted_2_scan_arg_11600, bool lifted_2_scan_arg_11601} =
    lifted_1_scan_7880(lifted_1_scan_arg_11599, 0i64)
  let {[n_11597]i64 s1_11602} =
    lifted_2_scan_7885(n_11597, lifted_2_scan_arg_11600,
                       lifted_2_scan_arg_11601, reps_11598)
  -- s1_11603 aliases s1_11602
  let {[n_11597]i64 s1_11603} = s1_11602
  let {i64 rotate_arg_11604} = sub64(0i64, 1i64)
  -- lifted_2_map2_arg_11605 aliases s1_11603
  let {[n_11597]i64 lifted_2_map2_arg_11605} =
    rotate_7751(n_11597, rotate_arg_11604, s1_11603)
  let {[n_11597]i64 lifted_1_map2_arg_11606} =
    iota_3662(n_11597)
  let {bool lifted_1_map2_arg_11607, bool lifted_1_map2_arg_11608} =
    lifted_0_map2_7886(true, true)
  -- lifted_2_map2_arg_11609 aliases lifted_1_map2_arg_11606
  let {[n_11597]i64 lifted_2_map2_arg_11609, bool lifted_2_map2_arg_11610,
       bool lifted_2_map2_arg_11611} =
    lifted_1_map2_7887(n_11597, lifted_1_map2_arg_11607,
                       lifted_1_map2_arg_11608, lifted_1_map2_arg_11606)
  let {[n_11597]i64 s2_11612} =
    lifted_2_map2_7894(n_11597, lifted_2_map2_arg_11609,
                       lifted_2_map2_arg_11610, lifted_2_map2_arg_11611,
                       lifted_2_map2_arg_11605)
  -- s2_11613 aliases s2_11612
  let {[n_11597]i64 s2_11613} = s2_11612
  let {[n_11597]i64 lifted_4_reduce_by_index_arg_11614} =
    iota_3662(n_11597)
  let {bool lifted_1_reduce_arg_11615} =
    lifted_0_reduce_7897(true, true)
  let {i64 lifted_2_reduce_arg_11616, bool lifted_2_reduce_arg_11617} =
    lifted_1_reduce_7898(lifted_1_reduce_arg_11615, 0i64)
  let {i64 replicate_arg_11618} =
    lifted_2_reduce_7903(n_11597, lifted_2_reduce_arg_11616,
                         lifted_2_reduce_arg_11617, reps_11598)
  let {i64 argdim₂₅_11619} = replicate_arg_11618
  let {[replicate_arg_11618]i64 reduce_by_index_arg_11620} =
    replicate_7755(replicate_arg_11618, 0i64)
  let {i64 size_11621;
       [size_11621]i64 lifted_1_reduce_by_index_arg_11622} =
    -- Consumes reduce_by_index_arg_11620
    reduce_by_index_7752(replicate_arg_11618, *reduce_by_index_arg_11620)
  let {[size_11621]i64 lifted_2_reduce_by_index_arg_11623,
       bool lifted_2_reduce_by_index_arg_11624} =
    -- Consumes lifted_1_reduce_by_index_arg_11622
    lifted_1_reduce_by_index_7904(size_11621,
                                  *lifted_1_reduce_by_index_arg_11622, true)
  let {[size_11621]i64 lifted_3_reduce_by_index_arg_11625,
       bool lifted_3_reduce_by_index_arg_11626,
       i64 lifted_3_reduce_by_index_arg_11627} =
    -- Consumes lifted_2_reduce_by_index_arg_11623
    lifted_2_reduce_by_index_7905(size_11621,
                                  *lifted_2_reduce_by_index_arg_11623,
                                  lifted_2_reduce_by_index_arg_11624, 0i64)
  -- lifted_4_reduce_by_index_arg_11630 aliases s2_11613
  let {[size_11621]i64 lifted_4_reduce_by_index_arg_11628,
       bool lifted_4_reduce_by_index_arg_11629,
       [n_11597]i64 lifted_4_reduce_by_index_arg_11630,
       i64 lifted_4_reduce_by_index_arg_11631} =
    -- Consumes lifted_3_reduce_by_index_arg_11625
    lifted_3_reduce_by_index_7906(n_11597, size_11621,
                                  *lifted_3_reduce_by_index_arg_11625,
                                  lifted_3_reduce_by_index_arg_11626,
                                  lifted_3_reduce_by_index_arg_11627, s2_11613)
  let {[size_11621]i64 tmp_11632} =
    -- Consumes lifted_4_reduce_by_index_arg_11628
    lifted_4_reduce_by_index_7911(size_11621, n_11597,
                                  *lifted_4_reduce_by_index_arg_11628,
                                  lifted_4_reduce_by_index_arg_11629,
                                  lifted_4_reduce_by_index_arg_11630,
                                  lifted_4_reduce_by_index_arg_11631,
                                  lifted_4_reduce_by_index_arg_11614)
  -- tmp_11633 aliases tmp_11632
  let {[size_11621]i64 tmp_11633} = tmp_11632
  let {bool lifted_1_map_arg_11634} =
    lifted_0_map_7912(true, true)
  let {[size_11621]bool flags_11635} =
    lifted_1_map_7915(size_11621, lifted_1_map_arg_11634, tmp_11633)
  -- flags_11636 aliases flags_11635
  let {[size_11621]bool flags_11636} = flags_11635
  let {bool lifted_1_segmented_scan_arg_11637,
       bool lifted_1_segmented_scan_arg_11638} =
    lifted_0_segmented_scan_7918(true, true)
  let {i64 lifted_2_segmented_scan_arg_11639,
       bool lifted_2_segmented_scan_arg_11640,
       bool lifted_2_segmented_scan_arg_11641} =
    lifted_1_segmented_scan_7919(lifted_1_segmented_scan_arg_11637,
                                 lifted_1_segmented_scan_arg_11638, 0i64)
  -- lifted_3_segmented_scan_arg_11642 aliases flags_11636
  let {[size_11621]bool lifted_3_segmented_scan_arg_11642,
       i64 lifted_3_segmented_scan_arg_11643,
       bool lifted_3_segmented_scan_arg_11644,
       bool lifted_3_segmented_scan_arg_11645} =
    lifted_2_segmented_scan_7920(size_11621, lifted_2_segmented_scan_arg_11639,
                                 lifted_2_segmented_scan_arg_11640,
                                 lifted_2_segmented_scan_arg_11641, flags_11636)
  -- res_11646 aliases tmp_11633, lifted_3_segmented_scan_arg_11642
  let {[size_11621]i64 res_11646} =
    lifted_3_segmented_scan_7930(size_11621, lifted_3_segmented_scan_arg_11642,
                                 lifted_3_segmented_scan_arg_11643,
                                 lifted_3_segmented_scan_arg_11644,
                                 lifted_3_segmented_scan_arg_11645, tmp_11633)
  in {size_11621, res_11646}
}

fun {i64} negate_1948 (i64 x_11647) = {
  let {i64 res_11648} = sub64(0i64, x_11647)
  in {res_11648}
}

fun {bool, bool} lifted_0_segmented_scan_7933 (bool scan_11649,
                                               bool op_11650) = {
  {op_11650, scan_11649}
}

fun {i64, bool, bool} lifted_1_segmented_scan_7934 (bool op_11651,
                                                    bool scan_11652,
                                                    i64 ne_11653) = {
  {ne_11653, op_11651, scan_11652}
}

fun {[n_11654]bool, i64, bool, bool} lifted_2_segmented_scan_7935 (i64 n_11654,
                                                                   i64 ne_11655,
                                                                   bool op_11656,
                                                                   bool scan_11657,
                                                                   [n_11654]bool flags_11658) = {
  {flags_11658, ne_11655, op_11656, scan_11657}
}

fun {bool} lifted_0_scan_7936 (bool nameless_11659, bool op_11660) = {
  {op_11660}
}

fun {bool, i64, bool} lifted_1_scan_7937 (bool op_11661, bool 0_11662,
                                          i64 1_11663) = {
  {0_11662, 1_11663, op_11661}
}

fun {bool, i64, bool} lifted_0_op_7940 (bool op_11664, bool x_flag_11665,
                                        i64 x_11666) = {
  {op_11664, x_11666, x_flag_11665}
}

fun {i64} lifted_0_op_7941 (bool nameless_11667, i64 x_11668) = {
  {x_11668}
}

fun {i64} lifted_1_op_7942 (i64 x_11669, i64 x_11670) = {
  let {i64 res_11671} = add64(x_11669, x_11670)
  in {res_11671}
}

fun {bool, i64} lifted_1_op_7943 (bool op_11672, i64 x_11673, bool x_flag_11674,
                                  bool y_flag_11675, i64 y_11676) = {
  let {bool res_11677} =
    -- Branch returns: {bool}
    if x_flag_11674
    then {true} else {y_flag_11675}
  let {i64 res_11678} =
    -- Branch returns: {i64}
    if y_flag_11675
    then {y_11676} else {
      let {i64 lifted_1_op_arg_11679} =
        lifted_0_op_7941(op_11672, x_11673)
      let {i64 res_11680} =
        lifted_1_op_7942(lifted_1_op_arg_11679, y_11676)
      in {res_11680}
    }
  in {res_11677, res_11678}
}

fun {*[n_11681]bool, *[n_11681]i64} lifted_2_scan_7944 (i64 n_11681,
                                                        bool 0_11682,
                                                        i64 1_11683,
                                                        bool op_11684,
                                                        [n_11681]bool as_11685,
                                                        [n_11681]i64 as_11686) = {
  let {[n_11681]bool res_11687, [n_11681]i64 res_11688} =
    scanomap(n_11681,
             {fn {bool, i64} (bool x_11689, i64 x_11690, bool x_11691,
                              i64 x_11692) =>
                let {bool lifted_1_op_arg_11693, i64 lifted_1_op_arg_11694,
                     bool lifted_1_op_arg_11695} =
                  lifted_0_op_7940(op_11684, x_11689, x_11690)
                let {bool res_11696, i64 res_11697} =
                  lifted_1_op_7943(lifted_1_op_arg_11693, lifted_1_op_arg_11694,
                                   lifted_1_op_arg_11695, x_11691, x_11692)
                in {res_11696, res_11697},
              {0_11682, 1_11683}},
             fn {bool, i64} (bool x_11698, i64 x_11699) =>
               {x_11698, x_11699},
             as_11685, as_11686)
  let {i64 ret₁_11700} = n_11681
  in {res_11687, res_11688}
}

fun {[n_11701]i64} lifted_3_segmented_scan_7945 (i64 n_11701,
                                                 [n_11701]bool flags_11702,
                                                 i64 ne_11703, bool op_11704,
                                                 bool scan_11705,
                                                 [n_11701]i64 as_11706) = {
  -- lifted_2_scan_arg_11707 aliases flags_11702, as_11706
  -- lifted_2_scan_arg_11708 aliases flags_11702, as_11706
  let {[n_11701]bool lifted_2_scan_arg_11707,
       [n_11701]i64 lifted_2_scan_arg_11708} =
    zip_7763(n_11701, flags_11702, as_11706)
  let {bool lifted_1_scan_arg_11709} =
    lifted_0_scan_7936(scan_11705, op_11704)
  let {bool lifted_2_scan_arg_11710, i64 lifted_2_scan_arg_11711,
       bool lifted_2_scan_arg_11712} =
    lifted_1_scan_7937(lifted_1_scan_arg_11709, false, ne_11703)
  let {[n_11701]bool unzip_arg_11713, [n_11701]i64 unzip_arg_11714} =
    lifted_2_scan_7944(n_11701, lifted_2_scan_arg_11710,
                       lifted_2_scan_arg_11711, lifted_2_scan_arg_11712,
                       lifted_2_scan_arg_11707, lifted_2_scan_arg_11708)
  -- res_11715 aliases unzip_arg_11713, unzip_arg_11714
  -- res_11716 aliases unzip_arg_11713, unzip_arg_11714
  let {[n_11701]bool res_11715, [n_11701]i64 res_11716} =
    unzip_7759(n_11701, unzip_arg_11713, unzip_arg_11714)
  in {res_11716}
}

fun {bool} lifted_0_map_7946 (bool nameless_11717, bool f_11718) = {
  {f_11718}
}

fun {i64} lifted_0_f_7948 (bool nameless_11719, i64 x_11720) = {
  let {i64 res_11721} = sub64(x_11720, 1i64)
  in {res_11721}
}

fun {*[n_11722]i64} lifted_1_map_7949 (i64 n_11722, bool f_11723,
                                       [n_11722]i64 as_11724) = {
  let {[n_11722]i64 res_11725} =
    map(n_11722,
        fn {i64} (i64 x_11726) =>
          let {i64 res_11727} =
            lifted_0_f_7948(f_11723, x_11726)
          in {res_11727},
        as_11724)
  let {i64 ret₂_11728} = n_11722
  in {res_11725}
}

fun {[n_11729]i64} segmented_iota_7768 (i64 n_11729,
                                        [n_11729]bool flags_11730) = {
  let {[n_11729]i64 lifted_3_segmented_scan_arg_11731} =
    replicate_7755(n_11729, 1i64)
  let {bool lifted_1_segmented_scan_arg_11732,
       bool lifted_1_segmented_scan_arg_11733} =
    lifted_0_segmented_scan_7933(true, true)
  let {i64 lifted_2_segmented_scan_arg_11734,
       bool lifted_2_segmented_scan_arg_11735,
       bool lifted_2_segmented_scan_arg_11736} =
    lifted_1_segmented_scan_7934(lifted_1_segmented_scan_arg_11732,
                                 lifted_1_segmented_scan_arg_11733, 0i64)
  -- lifted_3_segmented_scan_arg_11737 aliases flags_11730
  let {[n_11729]bool lifted_3_segmented_scan_arg_11737,
       i64 lifted_3_segmented_scan_arg_11738,
       bool lifted_3_segmented_scan_arg_11739,
       bool lifted_3_segmented_scan_arg_11740} =
    lifted_2_segmented_scan_7935(n_11729, lifted_2_segmented_scan_arg_11734,
                                 lifted_2_segmented_scan_arg_11735,
                                 lifted_2_segmented_scan_arg_11736, flags_11730)
  -- iotas_11741 aliases lifted_3_segmented_scan_arg_11731, lifted_3_segmented_scan_arg_11737
  let {[n_11729]i64 iotas_11741} =
    lifted_3_segmented_scan_7945(n_11729, lifted_3_segmented_scan_arg_11737,
                                 lifted_3_segmented_scan_arg_11738,
                                 lifted_3_segmented_scan_arg_11739,
                                 lifted_3_segmented_scan_arg_11740,
                                 lifted_3_segmented_scan_arg_11731)
  -- iotas_11742 aliases iotas_11741
  let {[n_11729]i64 iotas_11742} = iotas_11741
  let {bool lifted_1_map_arg_11743} =
    lifted_0_map_7946(true, true)
  let {[n_11729]i64 res_11744} =
    lifted_1_map_7949(n_11729, lifted_1_map_arg_11743, iotas_11742)
  in {res_11744}
}

fun {[n_11745]bool, [n_11745]f32} unzip_7771 (i64 n_11745,
                                              [n_11745]bool xs_11746,
                                              [n_11745]f32 xs_11747) = {
  let {i64 ret₂_11748} = n_11745
  let {i64 ret₃_11749} = n_11745
  in {xs_11746, xs_11747}
}

fun {[n_11750]bool, [n_11750]f32} zip_7775 (i64 n_11750, [n_11750]bool as_11751,
                                            [n_11750]f32 bs_11752) = {
  let {i64 ret₂_11753} = n_11750
  in {as_11751, bs_11752}
}

fun {[n_11754]bool} rotate_7777 (i64 n_11754, i64 r_11755,
                                 [n_11754]bool xs_11756) = {
  -- res_11757 aliases xs_11756
  let {[n_11754]bool res_11757} = rotate((r_11755), xs_11756)
  let {i64 ret₁_11758} = n_11754
  in {res_11757}
}

fun {i64, [?0]i64} |>_7780 (i64 d_11759, i64 d_11760, [d_11759]i64 x_11761) = {
  {d_11759, d_11760, x_11761}
}

fun {i64, [?0]bool} |>_7783 (i64 d_11762, i64 d_11763,
                             [d_11762]bool x_11764) = {
  {d_11762, d_11763, x_11764}
}

fun {i64} bool_1919 (bool x_11765) = {
  let {i64 res_11766} = btoi bool x_11765 to i64
  in {res_11766}
}

fun {i64} last_7789 (i64 n_11767, [n_11767]i64 x_11768) = {
  let {i64 i_11769} = sub64(n_11767, 1i64)
  let {bool x_11770} = sle64(0i64, i_11769)
  let {bool y_11771} = slt64(i_11769, n_11767)
  let {bool bounds_check_11772} = logand(x_11770, y_11771)
  let {cert index_certs_11773} =
    assert(bounds_check_11772, "Index [", i_11769,
                               "] out of bounds for array of shape [", n_11767,
                               "].", "/prelude/array.fut:18:29-34")
  let {i64 res_11774} =
    <index_certs_11773>
    x_11768[i_11769]
  in {res_11774}
}

fun {*[m_11775]f32} scatter_7790 (i64 m_11775, i64 n_11776,
                                  *[m_11775]f32 dest_11777,
                                  [n_11776]i64 is_11778,
                                  [n_11776]f32 vs_11779) = {
  let {bool write_cmp_11780} = eq_i64(n_11776, n_11776)
  let {cert write_cert_11781} =
    assert(write_cmp_11780, "length of index and value array does not match",
           "/prelude/soacs.fut:256:3-35")
  -- vs_write_sv_11782 aliases vs_11779
  let {[n_11776]f32 vs_write_sv_11782} =
    <write_cert_11781>
    reshape((~n_11776), vs_11779)
  let {[m_11775]f32 res_11783} =
    -- Consumes dest_11777
    scatter(n_11776,
            fn {i64, f32} (i64 write_index_11784, f32 write_value_11785) =>
              {write_index_11784, write_value_11785},
            {is_11778, vs_write_sv_11782}, (1, dest_11777))
  let {i64 ret₁_11786} = m_11775
  in {res_11783}
}

fun {[n_11787]i64, [n_11787]bool} zip_7792 (i64 n_11787, [n_11787]i64 as_11788,
                                            [n_11787]bool bs_11789) = {
  let {i64 ret₂_11790} = n_11787
  in {as_11788, bs_11789}
}

fun {[n_11791]i64, [n_11791]bool} zip2_7793 (i64 n_11791, [n_11791]i64 as_11792,
                                             [n_11791]bool bs_11793) = {
  -- res_11794 aliases as_11792, bs_11793
  -- res_11795 aliases as_11792, bs_11793
  let {[n_11791]i64 res_11794, [n_11791]bool res_11795} =
    zip_7792(n_11791, as_11792, bs_11793)
  in {res_11794, res_11795}
}

fun {f32} ceil_3349 (f32 x_11796) = {
  let {f32 res_11797} =
    ceil32(x_11796)
  in {res_11797}
}

fun {i64} f32_1916 (f32 x_11798) = {
  let {i64 res_11799} = fptosi f32 x_11798 to i64
  in {res_11799}
}

fun {i64} pricing_size_7298 (f32 r_11800, f32 swap_11801, f32 swap_11802,
                             i64 swap_11803, f32 swap_11804, f32 t_11805,
                             f32 vasicek_11806, f32 vasicek_11807,
                             f32 vasicek_11808, f32 vasicek_11809,
                             f32 vasicek_11810) = {
  let {f32 x_11811} = fdiv32(t_11805, swap_11804)
  let {f32 ceil_arg_11812} = fsub32(x_11811, 1.0f32)
  let {f32 payments_11813} =
    ceil_3349(ceil_arg_11812)
  let {f32 payments_11814} = payments_11813
  let {i64 y_11815} =
    f32_1916(payments_11814)
  let {i64 max_arg_11816} = sub64(swap_11803, y_11815)
  let {i64 res_11817} =
    max_1951(max_arg_11816, 0i64)
  in {res_11817}
}

fun {f32} coef_get_7327 (f32 r_11818, f32 swap_11819, f32 swap_11820,
                         i64 swap_11821, f32 swap_11822, f32 t_11823,
                         f32 vasicek_11824, f32 vasicek_11825,
                         f32 vasicek_11826, f32 vasicek_11827,
                         f32 vasicek_11828, i64 i_11829) = {
  let {i64 x_11830} =
    pricing_size_7298(r_11818, swap_11819, swap_11820, swap_11821, swap_11822,
                      t_11823, vasicek_11824, vasicek_11825, vasicek_11826,
                      vasicek_11827, vasicek_11828)
  let {i64 size_11831} = sub64(x_11830, 1i64)
  let {i64 size_11832} = size_11831
  let {f32 res_11833} = 0.0f32
  let {bool cond_11834} = eq_i64(i_11829, 0i64)
  let {f32 res_11835} =
    -- Branch returns: {f32}
    if cond_11834
    then {
      let {f32 res_11836} = fadd32(res_11833, 1.0f32)
      in {res_11836}
    } else {res_11833}
  let {f32 res_11837} = res_11835
  let {bool cond_11838} = slt64(0i64, i_11829)
  let {f32 res_11839} =
    -- Branch returns: {f32}
    if cond_11838
    then {
      let {f32 y_11840} = fmul32(swap_11819, swap_11822)
      let {f32 res_11841} = fsub32(res_11837, y_11840)
      in {res_11841}
    } else {res_11837}
  let {f32 res_11842} = res_11839
  let {bool cond_11843} = eq_i64(i_11829, size_11832)
  let {f32 res_11844} =
    -- Branch returns: {f32}
    if cond_11843
    then {
      let {f32 res_11845} = fsub32(res_11842, 1.0f32)
      in {res_11845}
    } else {res_11842}
  let {f32 res_11846} = res_11844
  let {f32 res_11847} = fmul32(swap_11820, res_11846)
  in {res_11847}
}

fun {f32} pricing_get_7348 (f32 r_11848, f32 swap_11849, f32 swap_11850,
                            i64 swap_11851, f32 swap_11852, f32 t_11853,
                            f32 vasicek_11854, f32 vasicek_11855,
                            f32 vasicek_11856, f32 vasicek_11857,
                            f32 vasicek_11858, i64 i_11859) = {
  let {f32 ceil_arg_11860} = fdiv32(t_11853, swap_11852)
  let {f32 x_11861} =
    ceil_3349(ceil_arg_11860)
  let {f32 start_11862} = fmul32(x_11861, swap_11852)
  let {f32 start_11863} = start_11862
  let {f32 coef_11864} =
    coef_get_7327(r_11848, swap_11849, swap_11850, swap_11851, swap_11852,
                  t_11853, vasicek_11854, vasicek_11855, vasicek_11856,
                  vasicek_11857, vasicek_11858, i_11859)
  let {f32 coef_11865} = coef_11864
  let {f32 x_11866} =
    i64_3244(i_11859)
  let {f32 y_11867} = fmul32(x_11866, swap_11852)
  let {f32 bondprice_arg_11868} = fadd32(start_11863, y_11867)
  let {f32 price_11869} =
    bondprice_7156(vasicek_11854, vasicek_11855, vasicek_11856, vasicek_11857,
                   vasicek_11858, r_11848, t_11853, bondprice_arg_11868)
  let {f32 price_11870} = price_11869
  let {f32 res_11871} = fmul32(coef_11865, price_11870)
  in {res_11871}
}

fun {[n_11874][m_11875][d_11873]f32} unflatten_7811 (i64 p_11872, i64 d_11873,
                                                     i64 n_11874, i64 m_11875,
                                                     [p_11872][d_11873]f32 xs_11876) = {
  let {i64 x_11877} = mul_nw64(n_11874, m_11875)
  let {bool dim_ok_11878} = eq_i64(x_11877, p_11872)
  let {cert dim_ok_cert_11879} =
    assert(dim_ok_11878,
           "new shape has different number of elements than old shape",
           "/prelude/array.fut:95:3-33")
  -- res_11880 aliases xs_11876
  let {[n_11874][m_11875][d_11873]f32 res_11880} =
    <dim_ok_cert_11879>
    reshape((n_11874, m_11875, d_11873), xs_11876)
  let {i64 ret₁_11881} = n_11874
  let {i64 ret₂_11882} = m_11875
  in {res_11880}
}

fun {[n_11884][m_11885]f32} unflatten_7812 (i64 p_11883, i64 n_11884,
                                            i64 m_11885,
                                            [p_11883]f32 xs_11886) = {
  let {i64 x_11887} = mul_nw64(n_11884, m_11885)
  let {bool dim_ok_11888} = eq_i64(x_11887, p_11883)
  let {cert dim_ok_cert_11889} =
    assert(dim_ok_11888,
           "new shape has different number of elements than old shape",
           "/prelude/array.fut:95:3-33")
  -- res_11890 aliases xs_11886
  let {[n_11884][m_11885]f32 res_11890} =
    <dim_ok_cert_11889>
    reshape((n_11884, m_11885), xs_11886)
  let {i64 ret₁_11891} = n_11884
  let {i64 ret₂_11892} = m_11885
  in {res_11890}
}

fun {[n_11894][m_11895][l_11896]f32} unflatten_3d_7813 (i64 p_11893,
                                                        i64 n_11894,
                                                        i64 m_11895,
                                                        i64 l_11896,
                                                        [p_11893]f32 xs_11897) = {
  let {i64 unflatten_arg_11898} = mul64(n_11894, m_11895)
  let {i64 argdim₅_11899} = unflatten_arg_11898
  -- unflatten_arg_11900 aliases xs_11897
  let {[unflatten_arg_11898][l_11896]f32 unflatten_arg_11900} =
    unflatten_7812(p_11893, unflatten_arg_11898, l_11896, xs_11897)
  -- res_11901 aliases unflatten_arg_11900
  let {[n_11894][m_11895][l_11896]f32 res_11901} =
    unflatten_7811(unflatten_arg_11898, l_11896, n_11894, m_11895,
                   unflatten_arg_11900)
  in {res_11901}
}

fun {[m_11903][n_11902][d_11904]f32} transpose_7815 (i64 n_11902, i64 m_11903,
                                                     i64 d_11904,
                                                     [n_11902][m_11903][d_11904]f32 a_11905) = {
  -- res_11906 aliases a_11905
  let {[m_11903][n_11902][d_11904]f32 res_11906} = rearrange((1, 0, 2), a_11905)
  let {i64 ret₁_11907} = m_11903
  let {i64 ret₂_11908} = n_11902
  in {res_11906}
}

fun {i64, i64} map_7818 (i64 d_11909, i64 d_11910) = {
  {d_11909, d_11910}
}

fun {i64} map_7820 (i64 d_11911) = {
  {d_11911}
}

fun {bool, bool} lifted_0_map2_7952 (bool map_11912, bool f_11913) = {
  {f_11913, map_11912}
}

fun {[n_11914]f32, bool, bool} lifted_1_map2_7953 (i64 n_11914, bool f_11915,
                                                   bool map_11916,
                                                   [n_11914]f32 as_11917) = {
  {as_11917, f_11915, map_11916}
}

fun {bool} lifted_0_map_7954 (bool nameless_11918, bool f_11919) = {
  {f_11919}
}

fun {f32} lifted_0_f_7956 (bool nameless_11920, f32 x_11921) = {
  {x_11921}
}

fun {f32} lifted_1_f_7957 (f32 x_11922, i64 y_11923) = {
  let {f32 y_11924} =
    i64_3244(y_11923)
  let {f32 res_11925} = fmul32(x_11922, y_11924)
  in {res_11925}
}

fun {f32} lifted_0_f_7958 (bool f_11926, f32 a_11927, i64 b_11928) = {
  let {f32 lifted_1_f_arg_11929} =
    lifted_0_f_7956(f_11926, a_11927)
  let {f32 res_11930} =
    lifted_1_f_7957(lifted_1_f_arg_11929, b_11928)
  in {res_11930}
}

fun {*[n_11931]f32} lifted_1_map_7959 (i64 n_11931, bool f_11932,
                                       [n_11931]f32 as_11933,
                                       [n_11931]i64 as_11934) = {
  let {[n_11931]f32 res_11935} =
    map(n_11931,
        fn {f32} (f32 x_11936, i64 x_11937) =>
          let {f32 res_11938} =
            lifted_0_f_7958(f_11932, x_11936, x_11937)
          in {res_11938},
        as_11933, as_11934)
  let {i64 ret₂_11939} = n_11931
  in {res_11935}
}

fun {*[n_11940]f32} lifted_2_map2_7960 (i64 n_11940, [n_11940]f32 as_11941,
                                        bool f_11942, bool map_11943,
                                        [n_11940]i64 bs_11944) = {
  -- lifted_1_map_arg_11945 aliases as_11941, bs_11944
  -- lifted_1_map_arg_11946 aliases as_11941, bs_11944
  let {[n_11940]f32 lifted_1_map_arg_11945,
       [n_11940]i64 lifted_1_map_arg_11946} =
    zip2_7681(n_11940, as_11941, bs_11944)
  let {bool lifted_1_map_arg_11947} =
    lifted_0_map_7954(map_11943, f_11942)
  let {[n_11940]f32 res_11948} =
    lifted_1_map_7959(n_11940, lifted_1_map_arg_11947, lifted_1_map_arg_11945,
                      lifted_1_map_arg_11946)
  in {res_11948}
}

fun {f32, f32, f32, [?0]f32, [?1]i64, f32, f32, [?2]f32}
lifted_0_map_7961 (i64 n_11949, bool nameless_11950, f32 f_11951, f32 f_11952,
                   f32 f_11953, [n_11949]f32 f_11954, [n_11949]i64 f_11955,
                   f32 f_11956, f32 f_11957, [n_11949]f32 f_11958) = {
  {n_11949, n_11949, n_11949, f_11951, f_11952, f_11953, f_11954, f_11955,
   f_11956, f_11957, f_11958}
}

fun {f32, f32, i64, f32} lifted_0_f_7963 (i64 n_11959, f32 a_11960, f32 b_11961,
                                          f32 deltat_11962,
                                          [n_11959]f32 notional_11963,
                                          [n_11959]i64 payments_11964,
                                          f32 r0_11965, f32 sigma_11966,
                                          [n_11959]f32 swap_term_11967,
                                          i64 x_11968) = {
  let {bool x_11969} = sle64(0i64, x_11968)
  let {bool y_11970} = slt64(x_11968, n_11959)
  let {bool bounds_check_11971} = logand(x_11969, y_11970)
  let {cert index_certs_11972} =
    assert(bounds_check_11971, "Index [", x_11968,
                               "] out of bounds for array of shape [", n_11959,
                               "].", "cva.fut:113:15-26")
  let {f32 res_11973} =
    <index_certs_11972>
    swap_term_11967[x_11968]
  let {bool x_11974} = sle64(0i64, x_11968)
  let {bool y_11975} = slt64(x_11968, n_11959)
  let {bool bounds_check_11976} = logand(x_11974, y_11975)
  let {cert index_certs_11977} =
    assert(bounds_check_11976, "Index [", x_11968,
                               "] out of bounds for array of shape [", n_11959,
                               "].", "cva.fut:114:18-28")
  let {i64 res_11978} =
    <index_certs_11977>
    payments_11964[x_11968]
  let {bool x_11979} = sle64(0i64, x_11968)
  let {bool y_11980} = slt64(x_11968, n_11959)
  let {bool bounds_check_11981} = logand(x_11979, y_11980)
  let {cert index_certs_11982} =
    assert(bounds_check_11981, "Index [", x_11968,
                               "] out of bounds for array of shape [", n_11959,
                               "].", "cva.fut:115:18-28")
  let {f32 res_11983} =
    <index_certs_11982>
    notional_11963[x_11968]
  let {bool x_11984} = sle64(0i64, x_11968)
  let {bool y_11985} = slt64(x_11968, n_11959)
  let {bool bounds_check_11986} = logand(x_11984, y_11985)
  let {cert index_certs_11987} =
    assert(bounds_check_11986, "Index [", x_11968,
                               "] out of bounds for array of shape [", n_11959,
                               "].", "cva.fut:116:44-54")
  let {i64 set_fixed_rate_arg_11988} =
    <index_certs_11987>
    payments_11964[x_11968]
  let {i64 argdim₁₈_11989} = set_fixed_rate_arg_11988
  let {bool x_11990} = sle64(0i64, x_11968)
  let {bool y_11991} = slt64(x_11968, n_11959)
  let {bool bounds_check_11992} = logand(x_11990, y_11991)
  let {cert index_certs_11993} =
    assert(bounds_check_11992, "Index [", x_11968,
                               "] out of bounds for array of shape [", n_11959,
                               "].", "cva.fut:116:31-42")
  let {f32 set_fixed_rate_arg_11994} =
    <index_certs_11993>
    swap_term_11967[x_11968]
  let {f32 res_11995} =
    set_fixed_rate_7400(set_fixed_rate_arg_11994, set_fixed_rate_arg_11988,
                        a_11960, b_11961, deltat_11962, r0_11965, sigma_11966)
  in {res_11995, res_11983, res_11978, res_11973}
}

fun {*[n_11996]f32, *[n_11996]f32, *[n_11996]i64, *[n_11996]f32}
lifted_1_map_7964 (i64 n_11996, i64 n_11997, f32 f_11998, f32 f_11999,
                   f32 f_12000, [n_11997]f32 f_12001, [n_11997]i64 f_12002,
                   f32 f_12003, f32 f_12004, [n_11997]f32 f_12005,
                   [n_11996]i64 as_12006) = {
  let {[n_11996]f32 res_12007, [n_11996]f32 res_12008, [n_11996]i64 res_12009,
       [n_11996]f32 res_12010} =
    map(n_11996,
        fn {f32, f32, i64, f32} (i64 x_12011) =>
          let {f32 res_12012, f32 res_12013, i64 res_12014, f32 res_12015} =
            lifted_0_f_7963(n_11997, f_11998, f_11999, f_12000, f_12001,
                            f_12002, f_12003, f_12004, f_12005, x_12011)
          in {res_12012, res_12013, res_12014, res_12015},
        as_12006)
  let {i64 ret₂_12016} = n_11996
  in {res_12007, res_12008, res_12009, res_12010}
}

fun {i64, bool, i64} lifted_1_map_7965 (i64 d_12017, bool f_12018,
                                        i64 f_12019) = {
  {d_12017, f_12018, f_12019}
}

fun {bool} lifted_0_map_7967 (bool nameless_12020, bool f_12021) = {
  {f_12021}
}

fun {f32} lifted_0_f_7969 (bool nameless_12022, i32 x_12023) = {
  let {i32 v_12024, f32 v_12025} =
    rand_7676(0.0f32, 1.0f32, x_12023)
  let {i32 nameless_12026} = v_12024
  let {f32 v_12027} = v_12025
  in {v_12027}
}

fun {*[n_12028]f32} lifted_1_map_7970 (i64 n_12028, bool f_12029,
                                       [n_12028]i32 as_12030) = {
  let {[n_12028]f32 res_12031} =
    map(n_12028,
        fn {f32} (i32 x_12032) =>
          let {f32 res_12033} =
            lifted_0_f_7969(f_12029, x_12032)
          in {res_12033},
        as_12030)
  let {i64 ret₂_12034} = n_12028
  in {res_12031}
}

fun {*[steps_12036]f32} lifted_0_f_7971 (bool map_12035, i64 steps_12036,
                                         i32 r_12037) = {
  let {[steps_12036]i32 rng_mat_12038} =
    split_rng_7575(steps_12036, r_12037)
  -- rng_mat_12039 aliases rng_mat_12038
  let {[steps_12036]i32 rng_mat_12039} = rng_mat_12038
  let {bool lifted_1_map_arg_12040} =
    lifted_0_map_7967(map_12035, true)
  let {[steps_12036]f32 row_12041} =
    lifted_1_map_7970(steps_12036, lifted_1_map_arg_12040, rng_mat_12039)
  -- row_12042 aliases row_12041
  let {[steps_12036]f32 row_12042} = row_12041
  in {row_12042}
}

fun {*[n_12043][d_12044]f32} lifted_2_map_7972 (i64 n_12043, i64 d_12044,
                                                bool f_12045, i64 f_12046,
                                                [n_12043]i32 as_12047) = {
  let {[n_12043][d_12044]f32 res_12048} =
    map(n_12043,
        fn {[d_12044]f32} (i32 x_12049) =>
          let {[f_12046]f32 res_12050} =
            lifted_0_f_7971(f_12045, f_12046, x_12049)
          let {bool dim_match_12051} = eq_i64(d_12044, f_12046)
          let {cert empty_or_match_cert_12052} =
            assert(dim_match_12051, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12053 aliases res_12050
          let {[d_12044]f32 result_proper_shape_12053} =
            <empty_or_match_cert_12052>
            reshape((~d_12044), res_12050)
          in {result_proper_shape_12053},
        as_12047)
  let {i64 ret₂_12054} = n_12043
  in {res_12048}
}

fun {i64, f32, f32, f32, f32, f32, f32} lifted_2_map_7973 (i64 d_12055,
                                                           i64 d_12056,
                                                           f32 f_12057,
                                                           f32 f_12058,
                                                           f32 f_12059,
                                                           f32 f_12060,
                                                           f32 f_12061,
                                                           f32 f_12062) = {
  {d_12056, f_12057, f_12058, f_12059, f_12060, f_12061, f_12062}
}

fun {[steps_12063]f32} lifted_0_f_7975 (i64 steps_12063, f32 a_12064,
                                        f32 b_12065, f32 deltat_12066,
                                        f32 r0_12067, f32 r0_12068,
                                        f32 sigma_12069,
                                        [steps_12063]f32 x_12070) = {
  -- res_12071 aliases x_12070
  let {[steps_12063]f32 res_12071} =
    mc_shortrate_7285(a_12064, b_12065, deltat_12066, r0_12068, sigma_12069,
                      r0_12067, steps_12063, x_12070)
  in {res_12071}
}

fun {*[n_12072][d_12074]f32} lifted_3_map_7976 (i64 n_12072, i64 d_12073,
                                                i64 d_12074, f32 f_12075,
                                                f32 f_12076, f32 f_12077,
                                                f32 f_12078, f32 f_12079,
                                                f32 f_12080,
                                                [n_12072][d_12073]f32 as_12081) = {
  let {[n_12072][d_12074]f32 res_12082} =
    map(n_12072,
        fn {[d_12074]f32} ([d_12073]f32 x_12083) =>
          -- res_12084 aliases x_12083
          let {[d_12073]f32 res_12084} =
            lifted_0_f_7975(d_12073, f_12075, f_12076, f_12077, f_12078,
                            f_12079, f_12080, x_12083)
          let {bool dim_match_12085} = eq_i64(d_12074, d_12073)
          let {cert empty_or_match_cert_12086} =
            assert(dim_match_12085, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12087 aliases res_12084
          let {[d_12074]f32 result_proper_shape_12087} =
            <empty_or_match_cert_12086>
            reshape((~d_12074), res_12084)
          in {result_proper_shape_12087},
        as_12081)
  let {i64 ret₂_12088} = n_12072
  in {res_12082}
}

fun {i64, i64, f32, f32, f32, bool, f32, f32, [?0]f32, [?0]f32, [?0]i64,
     [?0]f32, [?1]f32} lifted_3_map_7977 (i64 n_12089, i64 steps_12090,
                                          i64 d_12091, i64 d_12092, i64 d_12093,
                                          f32 f_12094, f32 f_12095, f32 f_12096,
                                          bool f_12097, f32 f_12098,
                                          f32 f_12099, [n_12089]f32 f_12100,
                                          [n_12089]f32 f_12101,
                                          [n_12089]i64 f_12102,
                                          [n_12089]f32 f_12103,
                                          [steps_12090]f32 f_12104) = {
  {n_12089, steps_12090, d_12092, d_12093, f_12094, f_12095, f_12096, f_12097,
   f_12098, f_12099, f_12100, f_12101, f_12102, f_12103, f_12104}
}

fun {i64, f32, f32, f32, bool, f32, f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32}
lifted_1_map2_7979 (i64 n_12105, i64 d_12106, f32 f_12107, f32 f_12108,
                    f32 f_12109, bool f_12110, f32 f_12111, f32 f_12112,
                    [n_12105]f32 f_12113, [n_12105]f32 f_12114,
                    [n_12105]i64 f_12115, [n_12105]f32 f_12116) = {
  {n_12105, d_12106, f_12107, f_12108, f_12109, f_12110, f_12111, f_12112,
   f_12113, f_12114, f_12115, f_12116}
}

fun {[n_12117]f32, i64, f32, f32, f32, bool, f32, f32, [?0]f32, [?0]f32,
     [?0]i64, [?0]f32} lifted_2_map2_7980 (i64 n_12117, i64 n_12118,
                                           i64 d_12119, f32 f_12120,
                                           f32 f_12121, f32 f_12122,
                                           bool f_12123, f32 f_12124,
                                           f32 f_12125, [n_12118]f32 f_12126,
                                           [n_12118]f32 f_12127,
                                           [n_12118]i64 f_12128,
                                           [n_12118]f32 f_12129,
                                           [n_12117]f32 as_12130) = {
  {n_12118, as_12130, d_12119, f_12120, f_12121, f_12122, f_12123, f_12124,
   f_12125, f_12126, f_12127, f_12128, f_12129}
}

fun {i64, f32, f32, f32, bool, f32, f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32}
lifted_1_map_7981 (i64 n_12131, i64 d_12132, f32 f_12133, f32 f_12134,
                   f32 f_12135, bool f_12136, f32 f_12137, f32 f_12138,
                   [n_12131]f32 f_12139, [n_12131]f32 f_12140,
                   [n_12131]i64 f_12141, [n_12131]f32 f_12142) = {
  {n_12131, d_12132, f_12133, f_12134, f_12135, f_12136, f_12137, f_12138,
   f_12139, f_12140, f_12141, f_12142}
}

fun {f32, f32, f32, bool, f32, f32, [n_12143]f32, [n_12143]f32, [n_12143]i64,
     [n_12143]f32, f32} lifted_0_f_7983 (i64 n_12143, f32 a_12144, f32 b_12145,
                                         f32 deltat_12146, bool map_12147,
                                         f32 r0_12148, f32 sigma_12149,
                                         [n_12143]f32 swaps_12150,
                                         [n_12143]f32 swaps_12151,
                                         [n_12143]i64 swaps_12152,
                                         [n_12143]f32 swaps_12153,
                                         f32 y_12154) = {
  {a_12144, b_12145, deltat_12146, map_12147, r0_12148, sigma_12149,
   swaps_12150, swaps_12151, swaps_12152, swaps_12153, y_12154}
}

fun {f32, f32, f32, f32, f32, f32, f32} lifted_0_map_7984 (bool nameless_12155,
                                                           f32 f_12156,
                                                           f32 f_12157,
                                                           f32 f_12158,
                                                           f32 f_12159,
                                                           f32 f_12160,
                                                           f32 f_12161,
                                                           f32 f_12162) = {
  {f_12156, f_12157, f_12158, f_12159, f_12160, f_12161, f_12162}
}

fun {f32, f32, f32, i64, f32, f32, f32, f32, f32, f32, f32}
lifted_0_f_7986 (f32 a_12163, f32 b_12164, f32 deltat_12165, f32 r0_12166,
                 f32 sigma_12167, f32 y_12168, f32 z_12169, f32 swap_12170,
                 f32 swap_12171, i64 swap_12172, f32 swap_12173) = {
  let {f32 r_12174} = y_12168
  let {f32 swap_12175} = swap_12170
  let {f32 swap_12176} = swap_12171
  let {i64 swap_12177} = swap_12172
  let {f32 swap_12178} = swap_12173
  let {f32 t_12179} = z_12169
  let {f32 vasicek_12180} = a_12163
  let {f32 vasicek_12181} = b_12164
  let {f32 vasicek_12182} = deltat_12165
  let {f32 vasicek_12183} = r0_12166
  let {f32 vasicek_12184} = sigma_12167
  in {r_12174, swap_12175, swap_12176, swap_12177, swap_12178, t_12179,
      vasicek_12180, vasicek_12181, vasicek_12182, vasicek_12183, vasicek_12184}
}

fun {*[n_12185]f32, *[n_12185]f32, *[n_12185]f32, *[n_12185]i64, *[n_12185]f32,
     *[n_12185]f32, *[n_12185]f32, *[n_12185]f32, *[n_12185]f32, *[n_12185]f32,
     *[n_12185]f32} lifted_1_map_7987 (i64 n_12185, f32 f_12186, f32 f_12187,
                                       f32 f_12188, f32 f_12189, f32 f_12190,
                                       f32 f_12191, f32 f_12192,
                                       [n_12185]f32 as_12193,
                                       [n_12185]f32 as_12194,
                                       [n_12185]i64 as_12195,
                                       [n_12185]f32 as_12196) = {
  let {[n_12185]f32 res_12197, [n_12185]f32 res_12198, [n_12185]f32 res_12199,
       [n_12185]i64 res_12200, [n_12185]f32 res_12201, [n_12185]f32 res_12202,
       [n_12185]f32 res_12203, [n_12185]f32 res_12204, [n_12185]f32 res_12205,
       [n_12185]f32 res_12206, [n_12185]f32 res_12207} =
    map(n_12185,
        fn {f32, f32, f32, i64, f32, f32, f32, f32, f32, f32, f32} (f32 x_12208,
                                                                    f32 x_12209,
                                                                    i64 x_12210,
                                                                    f32 x_12211) =>
          let {f32 res_12212, f32 res_12213, f32 res_12214, i64 res_12215,
               f32 res_12216, f32 res_12217, f32 res_12218, f32 res_12219,
               f32 res_12220, f32 res_12221, f32 res_12222} =
            lifted_0_f_7986(f_12186, f_12187, f_12188, f_12189, f_12190,
                            f_12191, f_12192, x_12208, x_12209, x_12210,
                            x_12211)
          in {res_12212, res_12213, res_12214, res_12215, res_12216, res_12217,
              res_12218, res_12219, res_12220, res_12221, res_12222},
        as_12193, as_12194, as_12195, as_12196)
  let {i64 ret₂_12223} = n_12185
  in {res_12197, res_12198, res_12199, res_12200, res_12201, res_12202,
      res_12203, res_12204, res_12205, res_12206, res_12207}
}

fun {*[n_12224]f32, *[n_12224]f32, *[n_12224]f32, *[n_12224]i64, *[n_12224]f32,
     *[n_12224]f32, *[n_12224]f32, *[n_12224]f32, *[n_12224]f32, *[n_12224]f32,
     *[n_12224]f32} lifted_1_f_7988 (i64 n_12224, f32 a_12225, f32 b_12226,
                                     f32 deltat_12227, bool map_12228,
                                     f32 r0_12229, f32 sigma_12230,
                                     [n_12224]f32 swaps_12231,
                                     [n_12224]f32 swaps_12232,
                                     [n_12224]i64 swaps_12233,
                                     [n_12224]f32 swaps_12234, f32 y_12235,
                                     f32 z_12236) = {
  let {f32 lifted_1_map_arg_12237, f32 lifted_1_map_arg_12238,
       f32 lifted_1_map_arg_12239, f32 lifted_1_map_arg_12240,
       f32 lifted_1_map_arg_12241, f32 lifted_1_map_arg_12242,
       f32 lifted_1_map_arg_12243} =
    lifted_0_map_7984(map_12228, a_12225, b_12226, deltat_12227, r0_12229,
                      sigma_12230, y_12235, z_12236)
  let {[n_12224]f32 res_12244, [n_12224]f32 res_12245, [n_12224]f32 res_12246,
       [n_12224]i64 res_12247, [n_12224]f32 res_12248, [n_12224]f32 res_12249,
       [n_12224]f32 res_12250, [n_12224]f32 res_12251, [n_12224]f32 res_12252,
       [n_12224]f32 res_12253, [n_12224]f32 res_12254} =
    lifted_1_map_7987(n_12224, lifted_1_map_arg_12237, lifted_1_map_arg_12238,
                      lifted_1_map_arg_12239, lifted_1_map_arg_12240,
                      lifted_1_map_arg_12241, lifted_1_map_arg_12242,
                      lifted_1_map_arg_12243, swaps_12231, swaps_12232,
                      swaps_12233, swaps_12234)
  in {res_12244, res_12245, res_12246, res_12247, res_12248, res_12249,
      res_12250, res_12251, res_12252, res_12253, res_12254}
}

fun {[?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32} lifted_0_f_7989 (i64 n_12255, f32 f_12256,
                                                 f32 f_12257, f32 f_12258,
                                                 bool f_12259, f32 f_12260,
                                                 f32 f_12261,
                                                 [n_12255]f32 f_12262,
                                                 [n_12255]f32 f_12263,
                                                 [n_12255]i64 f_12264,
                                                 [n_12255]f32 f_12265,
                                                 f32 a_12266, f32 b_12267) = {
  -- lifted_1_f_arg_12274 aliases f_12262, f_12263, f_12264, f_12265
  -- lifted_1_f_arg_12275 aliases f_12262, f_12263, f_12264, f_12265
  -- lifted_1_f_arg_12276 aliases f_12262, f_12263, f_12264, f_12265
  -- lifted_1_f_arg_12277 aliases f_12262, f_12263, f_12264, f_12265
  let {f32 lifted_1_f_arg_12268, f32 lifted_1_f_arg_12269,
       f32 lifted_1_f_arg_12270, bool lifted_1_f_arg_12271,
       f32 lifted_1_f_arg_12272, f32 lifted_1_f_arg_12273,
       [n_12255]f32 lifted_1_f_arg_12274, [n_12255]f32 lifted_1_f_arg_12275,
       [n_12255]i64 lifted_1_f_arg_12276, [n_12255]f32 lifted_1_f_arg_12277,
       f32 lifted_1_f_arg_12278} =
    lifted_0_f_7983(n_12255, f_12256, f_12257, f_12258, f_12259, f_12260,
                    f_12261, f_12262, f_12263, f_12264, f_12265, a_12266)
  let {[n_12255]f32 res_12279, [n_12255]f32 res_12280, [n_12255]f32 res_12281,
       [n_12255]i64 res_12282, [n_12255]f32 res_12283, [n_12255]f32 res_12284,
       [n_12255]f32 res_12285, [n_12255]f32 res_12286, [n_12255]f32 res_12287,
       [n_12255]f32 res_12288, [n_12255]f32 res_12289} =
    lifted_1_f_7988(n_12255, lifted_1_f_arg_12268, lifted_1_f_arg_12269,
                    lifted_1_f_arg_12270, lifted_1_f_arg_12271,
                    lifted_1_f_arg_12272, lifted_1_f_arg_12273,
                    lifted_1_f_arg_12274, lifted_1_f_arg_12275,
                    lifted_1_f_arg_12276, lifted_1_f_arg_12277,
                    lifted_1_f_arg_12278, b_12267)
  in {n_12255, res_12279, res_12280, res_12281, res_12282, res_12283, res_12284,
      res_12285, res_12286, res_12287, res_12288, res_12289}
}

fun {*[n_12290][d_12292]f32, *[n_12290][d_12292]f32, *[n_12290][d_12292]f32,
     *[n_12290][d_12292]i64, *[n_12290][d_12292]f32, *[n_12290][d_12292]f32,
     *[n_12290][d_12292]f32, *[n_12290][d_12292]f32, *[n_12290][d_12292]f32,
     *[n_12290][d_12292]f32, *[n_12290][d_12292]f32}
lifted_2_map_7990 (i64 n_12290, i64 n_12291, i64 d_12292, f32 f_12293,
                   f32 f_12294, f32 f_12295, bool f_12296, f32 f_12297,
                   f32 f_12298, [n_12291]f32 f_12299, [n_12291]f32 f_12300,
                   [n_12291]i64 f_12301, [n_12291]f32 f_12302,
                   [n_12290]f32 as_12303, [n_12290]f32 as_12304) = {
  let {[n_12290][d_12292]f32 res_12305, [n_12290][d_12292]f32 res_12306,
       [n_12290][d_12292]f32 res_12307, [n_12290][d_12292]i64 res_12308,
       [n_12290][d_12292]f32 res_12309, [n_12290][d_12292]f32 res_12310,
       [n_12290][d_12292]f32 res_12311, [n_12290][d_12292]f32 res_12312,
       [n_12290][d_12292]f32 res_12313, [n_12290][d_12292]f32 res_12314,
       [n_12290][d_12292]f32 res_12315} =
    map(n_12290,
        fn {[d_12292]f32, [d_12292]f32, [d_12292]f32, [d_12292]i64,
            [d_12292]f32, [d_12292]f32, [d_12292]f32, [d_12292]f32,
            [d_12292]f32, [d_12292]f32, [d_12292]f32} (f32 x_12316,
                                                       f32 x_12317) =>
          -- res_12319 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12320 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12321 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12322 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12323 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12324 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12325 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12326 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12327 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12328 aliases f_12299, f_12300, f_12301, f_12302
          -- res_12329 aliases f_12299, f_12300, f_12301, f_12302
          let {i64 size_12318;
               [size_12318]f32 res_12319, [size_12318]f32 res_12320,
               [size_12318]f32 res_12321, [size_12318]i64 res_12322,
               [size_12318]f32 res_12323, [size_12318]f32 res_12324,
               [size_12318]f32 res_12325, [size_12318]f32 res_12326,
               [size_12318]f32 res_12327, [size_12318]f32 res_12328,
               [size_12318]f32 res_12329} =
            lifted_0_f_7989(n_12291, f_12293, f_12294, f_12295, f_12296,
                            f_12297, f_12298, f_12299, f_12300, f_12301,
                            f_12302, x_12316, x_12317)
          let {bool dim_match_12330} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12331} =
            assert(dim_match_12330, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12332 aliases res_12319
          let {[d_12292]f32 result_proper_shape_12332} =
            <empty_or_match_cert_12331>
            reshape((~d_12292), res_12319)
          let {bool dim_match_12333} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12334} =
            assert(dim_match_12333, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12335 aliases res_12320
          let {[d_12292]f32 result_proper_shape_12335} =
            <empty_or_match_cert_12334>
            reshape((~d_12292), res_12320)
          let {bool dim_match_12336} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12337} =
            assert(dim_match_12336, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12338 aliases res_12321
          let {[d_12292]f32 result_proper_shape_12338} =
            <empty_or_match_cert_12337>
            reshape((~d_12292), res_12321)
          let {bool dim_match_12339} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12340} =
            assert(dim_match_12339, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12341 aliases res_12322
          let {[d_12292]i64 result_proper_shape_12341} =
            <empty_or_match_cert_12340>
            reshape((~d_12292), res_12322)
          let {bool dim_match_12342} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12343} =
            assert(dim_match_12342, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12344 aliases res_12323
          let {[d_12292]f32 result_proper_shape_12344} =
            <empty_or_match_cert_12343>
            reshape((~d_12292), res_12323)
          let {bool dim_match_12345} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12346} =
            assert(dim_match_12345, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12347 aliases res_12324
          let {[d_12292]f32 result_proper_shape_12347} =
            <empty_or_match_cert_12346>
            reshape((~d_12292), res_12324)
          let {bool dim_match_12348} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12349} =
            assert(dim_match_12348, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12350 aliases res_12325
          let {[d_12292]f32 result_proper_shape_12350} =
            <empty_or_match_cert_12349>
            reshape((~d_12292), res_12325)
          let {bool dim_match_12351} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12352} =
            assert(dim_match_12351, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12353 aliases res_12326
          let {[d_12292]f32 result_proper_shape_12353} =
            <empty_or_match_cert_12352>
            reshape((~d_12292), res_12326)
          let {bool dim_match_12354} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12355} =
            assert(dim_match_12354, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12356 aliases res_12327
          let {[d_12292]f32 result_proper_shape_12356} =
            <empty_or_match_cert_12355>
            reshape((~d_12292), res_12327)
          let {bool dim_match_12357} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12358} =
            assert(dim_match_12357, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12359 aliases res_12328
          let {[d_12292]f32 result_proper_shape_12359} =
            <empty_or_match_cert_12358>
            reshape((~d_12292), res_12328)
          let {bool dim_match_12360} = eq_i64(d_12292, size_12318)
          let {cert empty_or_match_cert_12361} =
            assert(dim_match_12360, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12362 aliases res_12329
          let {[d_12292]f32 result_proper_shape_12362} =
            <empty_or_match_cert_12361>
            reshape((~d_12292), res_12329)
          in {result_proper_shape_12332, result_proper_shape_12335,
              result_proper_shape_12338, result_proper_shape_12341,
              result_proper_shape_12344, result_proper_shape_12347,
              result_proper_shape_12350, result_proper_shape_12353,
              result_proper_shape_12356, result_proper_shape_12359,
              result_proper_shape_12362},
        as_12303, as_12304)
  let {i64 ret₂_12363} = n_12290
  in {res_12305, res_12306, res_12307, res_12308, res_12309, res_12310,
      res_12311, res_12312, res_12313, res_12314, res_12315}
}

fun {*[n_12364][d_12367]f32, *[n_12364][d_12367]f32, *[n_12364][d_12367]f32,
     *[n_12364][d_12367]i64, *[n_12364][d_12367]f32, *[n_12364][d_12367]f32,
     *[n_12364][d_12367]f32, *[n_12364][d_12367]f32, *[n_12364][d_12367]f32,
     *[n_12364][d_12367]f32, *[n_12364][d_12367]f32}
lifted_3_map2_7991 (i64 n_12364, i64 n_12365, [n_12364]f32 as_12366,
                    i64 d_12367, f32 f_12368, f32 f_12369, f32 f_12370,
                    bool f_12371, f32 f_12372, f32 f_12373,
                    [n_12365]f32 f_12374, [n_12365]f32 f_12375,
                    [n_12365]i64 f_12376, [n_12365]f32 f_12377,
                    [n_12364]f32 bs_12378) = {
  -- lifted_2_map_arg_12379 aliases as_12366, bs_12378
  -- lifted_2_map_arg_12380 aliases as_12366, bs_12378
  let {[n_12364]f32 lifted_2_map_arg_12379,
       [n_12364]f32 lifted_2_map_arg_12380} =
    zip2_7734(n_12364, as_12366, bs_12378)
  let {i64 lifted_1_map_arg_12381} =
    map_7732(d_12367)
  -- lifted_2_map_arg_12390 aliases f_12374, f_12375, f_12376, f_12377
  -- lifted_2_map_arg_12391 aliases f_12374, f_12375, f_12376, f_12377
  -- lifted_2_map_arg_12392 aliases f_12374, f_12375, f_12376, f_12377
  -- lifted_2_map_arg_12393 aliases f_12374, f_12375, f_12376, f_12377
  let {i64 size_12382;
       i64 lifted_2_map_arg_12383, f32 lifted_2_map_arg_12384,
       f32 lifted_2_map_arg_12385, f32 lifted_2_map_arg_12386,
       bool lifted_2_map_arg_12387, f32 lifted_2_map_arg_12388,
       f32 lifted_2_map_arg_12389, [size_12382]f32 lifted_2_map_arg_12390,
       [size_12382]f32 lifted_2_map_arg_12391,
       [size_12382]i64 lifted_2_map_arg_12392,
       [size_12382]f32 lifted_2_map_arg_12393} =
    lifted_1_map_7981(n_12365, lifted_1_map_arg_12381, f_12368, f_12369,
                      f_12370, f_12371, f_12372, f_12373, f_12374, f_12375,
                      f_12376, f_12377)
  let {[n_12364][lifted_2_map_arg_12383]f32 res_12394,
       [n_12364][lifted_2_map_arg_12383]f32 res_12395,
       [n_12364][lifted_2_map_arg_12383]f32 res_12396,
       [n_12364][lifted_2_map_arg_12383]i64 res_12397,
       [n_12364][lifted_2_map_arg_12383]f32 res_12398,
       [n_12364][lifted_2_map_arg_12383]f32 res_12399,
       [n_12364][lifted_2_map_arg_12383]f32 res_12400,
       [n_12364][lifted_2_map_arg_12383]f32 res_12401,
       [n_12364][lifted_2_map_arg_12383]f32 res_12402,
       [n_12364][lifted_2_map_arg_12383]f32 res_12403,
       [n_12364][lifted_2_map_arg_12383]f32 res_12404} =
    lifted_2_map_7990(n_12364, size_12382, lifted_2_map_arg_12383,
                      lifted_2_map_arg_12384, lifted_2_map_arg_12385,
                      lifted_2_map_arg_12386, lifted_2_map_arg_12387,
                      lifted_2_map_arg_12388, lifted_2_map_arg_12389,
                      lifted_2_map_arg_12390, lifted_2_map_arg_12391,
                      lifted_2_map_arg_12392, lifted_2_map_arg_12393,
                      lifted_2_map_arg_12379, lifted_2_map_arg_12380)
  let {bool dim_match_12405} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12406} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12407} = logand(dim_match_12406, dim_match_12405)
  let {cert empty_or_match_cert_12408} =
    assert(match_12407,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12409 aliases res_12394
  let {[n_12364][d_12367]f32 result_proper_shape_12409} =
    <empty_or_match_cert_12408>
    reshape((~n_12364, ~d_12367), res_12394)
  let {bool dim_match_12410} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12411} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12412} = logand(dim_match_12411, dim_match_12410)
  let {cert empty_or_match_cert_12413} =
    assert(match_12412,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12414 aliases res_12395
  let {[n_12364][d_12367]f32 result_proper_shape_12414} =
    <empty_or_match_cert_12413>
    reshape((~n_12364, ~d_12367), res_12395)
  let {bool dim_match_12415} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12416} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12417} = logand(dim_match_12416, dim_match_12415)
  let {cert empty_or_match_cert_12418} =
    assert(match_12417,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12419 aliases res_12396
  let {[n_12364][d_12367]f32 result_proper_shape_12419} =
    <empty_or_match_cert_12418>
    reshape((~n_12364, ~d_12367), res_12396)
  let {bool dim_match_12420} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12421} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12422} = logand(dim_match_12421, dim_match_12420)
  let {cert empty_or_match_cert_12423} =
    assert(match_12422,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12424 aliases res_12397
  let {[n_12364][d_12367]i64 result_proper_shape_12424} =
    <empty_or_match_cert_12423>
    reshape((~n_12364, ~d_12367), res_12397)
  let {bool dim_match_12425} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12426} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12427} = logand(dim_match_12426, dim_match_12425)
  let {cert empty_or_match_cert_12428} =
    assert(match_12427,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12429 aliases res_12398
  let {[n_12364][d_12367]f32 result_proper_shape_12429} =
    <empty_or_match_cert_12428>
    reshape((~n_12364, ~d_12367), res_12398)
  let {bool dim_match_12430} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12431} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12432} = logand(dim_match_12431, dim_match_12430)
  let {cert empty_or_match_cert_12433} =
    assert(match_12432,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12434 aliases res_12399
  let {[n_12364][d_12367]f32 result_proper_shape_12434} =
    <empty_or_match_cert_12433>
    reshape((~n_12364, ~d_12367), res_12399)
  let {bool dim_match_12435} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12436} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12437} = logand(dim_match_12436, dim_match_12435)
  let {cert empty_or_match_cert_12438} =
    assert(match_12437,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12439 aliases res_12400
  let {[n_12364][d_12367]f32 result_proper_shape_12439} =
    <empty_or_match_cert_12438>
    reshape((~n_12364, ~d_12367), res_12400)
  let {bool dim_match_12440} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12441} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12442} = logand(dim_match_12441, dim_match_12440)
  let {cert empty_or_match_cert_12443} =
    assert(match_12442,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12444 aliases res_12401
  let {[n_12364][d_12367]f32 result_proper_shape_12444} =
    <empty_or_match_cert_12443>
    reshape((~n_12364, ~d_12367), res_12401)
  let {bool dim_match_12445} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12446} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12447} = logand(dim_match_12446, dim_match_12445)
  let {cert empty_or_match_cert_12448} =
    assert(match_12447,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12449 aliases res_12402
  let {[n_12364][d_12367]f32 result_proper_shape_12449} =
    <empty_or_match_cert_12448>
    reshape((~n_12364, ~d_12367), res_12402)
  let {bool dim_match_12450} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12451} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12452} = logand(dim_match_12451, dim_match_12450)
  let {cert empty_or_match_cert_12453} =
    assert(match_12452,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12454 aliases res_12403
  let {[n_12364][d_12367]f32 result_proper_shape_12454} =
    <empty_or_match_cert_12453>
    reshape((~n_12364, ~d_12367), res_12403)
  let {bool dim_match_12455} = eq_i64(n_12364, n_12364)
  let {bool dim_match_12456} = eq_i64(d_12367, lifted_2_map_arg_12383)
  let {bool match_12457} = logand(dim_match_12456, dim_match_12455)
  let {cert empty_or_match_cert_12458} =
    assert(match_12457,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12459 aliases res_12404
  let {[n_12364][d_12367]f32 result_proper_shape_12459} =
    <empty_or_match_cert_12458>
    reshape((~n_12364, ~d_12367), res_12404)
  in {result_proper_shape_12409, result_proper_shape_12414,
      result_proper_shape_12419, result_proper_shape_12424,
      result_proper_shape_12429, result_proper_shape_12434,
      result_proper_shape_12439, result_proper_shape_12444,
      result_proper_shape_12449, result_proper_shape_12454,
      result_proper_shape_12459}
}

fun {*[steps_12461][n_12460]f32, *[steps_12461][n_12460]f32,
     *[steps_12461][n_12460]f32, *[steps_12461][n_12460]i64,
     *[steps_12461][n_12460]f32, *[steps_12461][n_12460]f32,
     *[steps_12461][n_12460]f32, *[steps_12461][n_12460]f32,
     *[steps_12461][n_12460]f32, *[steps_12461][n_12460]f32,
     *[steps_12461][n_12460]f32} lifted_0_f_7992 (i64 n_12460, i64 steps_12461,
                                                  f32 a_12462, f32 b_12463,
                                                  f32 deltat_12464,
                                                  bool map_12465, f32 r0_12466,
                                                  f32 sigma_12467,
                                                  [n_12460]f32 swaps_12468,
                                                  [n_12460]f32 swaps_12469,
                                                  [n_12460]i64 swaps_12470,
                                                  [n_12460]f32 swaps_12471,
                                                  [steps_12461]f32 times_12472,
                                                  [steps_12461]f32 x_12473) = {
  let {i64 lifted_1_map2_arg_12474} =
    map2_7735(n_12460)
  -- lifted_2_map2_arg_12483 aliases swaps_12468, swaps_12469, swaps_12470, swaps_12471
  -- lifted_2_map2_arg_12484 aliases swaps_12468, swaps_12469, swaps_12470, swaps_12471
  -- lifted_2_map2_arg_12485 aliases swaps_12468, swaps_12469, swaps_12470, swaps_12471
  -- lifted_2_map2_arg_12486 aliases swaps_12468, swaps_12469, swaps_12470, swaps_12471
  let {i64 size_12475;
       i64 lifted_2_map2_arg_12476, f32 lifted_2_map2_arg_12477,
       f32 lifted_2_map2_arg_12478, f32 lifted_2_map2_arg_12479,
       bool lifted_2_map2_arg_12480, f32 lifted_2_map2_arg_12481,
       f32 lifted_2_map2_arg_12482, [size_12475]f32 lifted_2_map2_arg_12483,
       [size_12475]f32 lifted_2_map2_arg_12484,
       [size_12475]i64 lifted_2_map2_arg_12485,
       [size_12475]f32 lifted_2_map2_arg_12486} =
    lifted_1_map2_7979(n_12460, lifted_1_map2_arg_12474, a_12462, b_12463,
                       deltat_12464, map_12465, r0_12466, sigma_12467,
                       swaps_12468, swaps_12469, swaps_12470, swaps_12471)
  -- lifted_3_map2_arg_12488 aliases x_12473, lifted_2_map2_arg_12483, lifted_2_map2_arg_12484, lifted_2_map2_arg_12485, lifted_2_map2_arg_12486
  -- lifted_3_map2_arg_12496 aliases x_12473, lifted_2_map2_arg_12483, lifted_2_map2_arg_12484, lifted_2_map2_arg_12485, lifted_2_map2_arg_12486
  -- lifted_3_map2_arg_12497 aliases x_12473, lifted_2_map2_arg_12483, lifted_2_map2_arg_12484, lifted_2_map2_arg_12485, lifted_2_map2_arg_12486
  -- lifted_3_map2_arg_12498 aliases x_12473, lifted_2_map2_arg_12483, lifted_2_map2_arg_12484, lifted_2_map2_arg_12485, lifted_2_map2_arg_12486
  -- lifted_3_map2_arg_12499 aliases x_12473, lifted_2_map2_arg_12483, lifted_2_map2_arg_12484, lifted_2_map2_arg_12485, lifted_2_map2_arg_12486
  let {i64 size_12487;
       [steps_12461]f32 lifted_3_map2_arg_12488, i64 lifted_3_map2_arg_12489,
       f32 lifted_3_map2_arg_12490, f32 lifted_3_map2_arg_12491,
       f32 lifted_3_map2_arg_12492, bool lifted_3_map2_arg_12493,
       f32 lifted_3_map2_arg_12494, f32 lifted_3_map2_arg_12495,
       [size_12487]f32 lifted_3_map2_arg_12496,
       [size_12487]f32 lifted_3_map2_arg_12497,
       [size_12487]i64 lifted_3_map2_arg_12498,
       [size_12487]f32 lifted_3_map2_arg_12499} =
    lifted_2_map2_7980(steps_12461, size_12475, lifted_2_map2_arg_12476,
                       lifted_2_map2_arg_12477, lifted_2_map2_arg_12478,
                       lifted_2_map2_arg_12479, lifted_2_map2_arg_12480,
                       lifted_2_map2_arg_12481, lifted_2_map2_arg_12482,
                       lifted_2_map2_arg_12483, lifted_2_map2_arg_12484,
                       lifted_2_map2_arg_12485, lifted_2_map2_arg_12486,
                       x_12473)
  let {[steps_12461][lifted_3_map2_arg_12489]f32 res_12500,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12501,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12502,
       [steps_12461][lifted_3_map2_arg_12489]i64 res_12503,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12504,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12505,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12506,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12507,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12508,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12509,
       [steps_12461][lifted_3_map2_arg_12489]f32 res_12510} =
    lifted_3_map2_7991(steps_12461, size_12487, lifted_3_map2_arg_12488,
                       lifted_3_map2_arg_12489, lifted_3_map2_arg_12490,
                       lifted_3_map2_arg_12491, lifted_3_map2_arg_12492,
                       lifted_3_map2_arg_12493, lifted_3_map2_arg_12494,
                       lifted_3_map2_arg_12495, lifted_3_map2_arg_12496,
                       lifted_3_map2_arg_12497, lifted_3_map2_arg_12498,
                       lifted_3_map2_arg_12499, times_12472)
  let {bool dim_match_12511} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12512} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12513} = logand(dim_match_12512, dim_match_12511)
  let {cert empty_or_match_cert_12514} =
    assert(match_12513,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12515 aliases res_12500
  let {[steps_12461][n_12460]f32 result_proper_shape_12515} =
    <empty_or_match_cert_12514>
    reshape((~steps_12461, ~n_12460), res_12500)
  let {bool dim_match_12516} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12517} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12518} = logand(dim_match_12517, dim_match_12516)
  let {cert empty_or_match_cert_12519} =
    assert(match_12518,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12520 aliases res_12501
  let {[steps_12461][n_12460]f32 result_proper_shape_12520} =
    <empty_or_match_cert_12519>
    reshape((~steps_12461, ~n_12460), res_12501)
  let {bool dim_match_12521} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12522} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12523} = logand(dim_match_12522, dim_match_12521)
  let {cert empty_or_match_cert_12524} =
    assert(match_12523,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12525 aliases res_12502
  let {[steps_12461][n_12460]f32 result_proper_shape_12525} =
    <empty_or_match_cert_12524>
    reshape((~steps_12461, ~n_12460), res_12502)
  let {bool dim_match_12526} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12527} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12528} = logand(dim_match_12527, dim_match_12526)
  let {cert empty_or_match_cert_12529} =
    assert(match_12528,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12530 aliases res_12503
  let {[steps_12461][n_12460]i64 result_proper_shape_12530} =
    <empty_or_match_cert_12529>
    reshape((~steps_12461, ~n_12460), res_12503)
  let {bool dim_match_12531} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12532} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12533} = logand(dim_match_12532, dim_match_12531)
  let {cert empty_or_match_cert_12534} =
    assert(match_12533,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12535 aliases res_12504
  let {[steps_12461][n_12460]f32 result_proper_shape_12535} =
    <empty_or_match_cert_12534>
    reshape((~steps_12461, ~n_12460), res_12504)
  let {bool dim_match_12536} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12537} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12538} = logand(dim_match_12537, dim_match_12536)
  let {cert empty_or_match_cert_12539} =
    assert(match_12538,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12540 aliases res_12505
  let {[steps_12461][n_12460]f32 result_proper_shape_12540} =
    <empty_or_match_cert_12539>
    reshape((~steps_12461, ~n_12460), res_12505)
  let {bool dim_match_12541} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12542} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12543} = logand(dim_match_12542, dim_match_12541)
  let {cert empty_or_match_cert_12544} =
    assert(match_12543,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12545 aliases res_12506
  let {[steps_12461][n_12460]f32 result_proper_shape_12545} =
    <empty_or_match_cert_12544>
    reshape((~steps_12461, ~n_12460), res_12506)
  let {bool dim_match_12546} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12547} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12548} = logand(dim_match_12547, dim_match_12546)
  let {cert empty_or_match_cert_12549} =
    assert(match_12548,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12550 aliases res_12507
  let {[steps_12461][n_12460]f32 result_proper_shape_12550} =
    <empty_or_match_cert_12549>
    reshape((~steps_12461, ~n_12460), res_12507)
  let {bool dim_match_12551} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12552} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12553} = logand(dim_match_12552, dim_match_12551)
  let {cert empty_or_match_cert_12554} =
    assert(match_12553,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12555 aliases res_12508
  let {[steps_12461][n_12460]f32 result_proper_shape_12555} =
    <empty_or_match_cert_12554>
    reshape((~steps_12461, ~n_12460), res_12508)
  let {bool dim_match_12556} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12557} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12558} = logand(dim_match_12557, dim_match_12556)
  let {cert empty_or_match_cert_12559} =
    assert(match_12558,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12560 aliases res_12509
  let {[steps_12461][n_12460]f32 result_proper_shape_12560} =
    <empty_or_match_cert_12559>
    reshape((~steps_12461, ~n_12460), res_12509)
  let {bool dim_match_12561} = eq_i64(steps_12461, steps_12461)
  let {bool dim_match_12562} = eq_i64(n_12460, lifted_3_map2_arg_12489)
  let {bool match_12563} = logand(dim_match_12562, dim_match_12561)
  let {cert empty_or_match_cert_12564} =
    assert(match_12563,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_12565 aliases res_12510
  let {[steps_12461][n_12460]f32 result_proper_shape_12565} =
    <empty_or_match_cert_12564>
    reshape((~steps_12461, ~n_12460), res_12510)
  in {result_proper_shape_12515, result_proper_shape_12520,
      result_proper_shape_12525, result_proper_shape_12530,
      result_proper_shape_12535, result_proper_shape_12540,
      result_proper_shape_12545, result_proper_shape_12550,
      result_proper_shape_12555, result_proper_shape_12560,
      result_proper_shape_12565}
}

fun {*[n_12566][d_12570][d_12571]f32, *[n_12566][d_12570][d_12571]f32,
     *[n_12566][d_12570][d_12571]f32, *[n_12566][d_12570][d_12571]i64,
     *[n_12566][d_12570][d_12571]f32, *[n_12566][d_12570][d_12571]f32,
     *[n_12566][d_12570][d_12571]f32, *[n_12566][d_12570][d_12571]f32,
     *[n_12566][d_12570][d_12571]f32, *[n_12566][d_12570][d_12571]f32,
     *[n_12566][d_12570][d_12571]f32} lifted_4_map_7993 (i64 n_12566,
                                                         i64 d_12567,
                                                         i64 n_12568,
                                                         i64 steps_12569,
                                                         i64 d_12570,
                                                         i64 d_12571,
                                                         f32 f_12572,
                                                         f32 f_12573,
                                                         f32 f_12574,
                                                         bool f_12575,
                                                         f32 f_12576,
                                                         f32 f_12577,
                                                         [n_12568]f32 f_12578,
                                                         [n_12568]f32 f_12579,
                                                         [n_12568]i64 f_12580,
                                                         [n_12568]f32 f_12581,
                                                         [steps_12569]f32 f_12582,
                                                         [n_12566][d_12567]f32 as_12583) = {
  let {[n_12566][d_12570][d_12571]f32 res_12584,
       [n_12566][d_12570][d_12571]f32 res_12585,
       [n_12566][d_12570][d_12571]f32 res_12586,
       [n_12566][d_12570][d_12571]i64 res_12587,
       [n_12566][d_12570][d_12571]f32 res_12588,
       [n_12566][d_12570][d_12571]f32 res_12589,
       [n_12566][d_12570][d_12571]f32 res_12590,
       [n_12566][d_12570][d_12571]f32 res_12591,
       [n_12566][d_12570][d_12571]f32 res_12592,
       [n_12566][d_12570][d_12571]f32 res_12593,
       [n_12566][d_12570][d_12571]f32 res_12594} =
    map(n_12566,
        fn {[d_12570][d_12571]f32, [d_12570][d_12571]f32, [d_12570][d_12571]f32,
            [d_12570][d_12571]i64, [d_12570][d_12571]f32, [d_12570][d_12571]f32,
            [d_12570][d_12571]f32, [d_12570][d_12571]f32, [d_12570][d_12571]f32,
            [d_12570][d_12571]f32, [d_12570][d_12571]f32}
        ([d_12567]f32 x_12595) =>
          let {bool dim_match_12596} = eq_i64(steps_12569, d_12567)
          let {cert empty_or_match_cert_12597} =
            assert(dim_match_12596, "function arguments of wrong shape",
                   "unknown location")
          -- x_12598 aliases x_12595
          let {[steps_12569]f32 x_12598} =
            <empty_or_match_cert_12597>
            reshape((~steps_12569), x_12595)
          let {[steps_12569][n_12568]f32 res_12599,
               [steps_12569][n_12568]f32 res_12600,
               [steps_12569][n_12568]f32 res_12601,
               [steps_12569][n_12568]i64 res_12602,
               [steps_12569][n_12568]f32 res_12603,
               [steps_12569][n_12568]f32 res_12604,
               [steps_12569][n_12568]f32 res_12605,
               [steps_12569][n_12568]f32 res_12606,
               [steps_12569][n_12568]f32 res_12607,
               [steps_12569][n_12568]f32 res_12608,
               [steps_12569][n_12568]f32 res_12609} =
            lifted_0_f_7992(n_12568, steps_12569, f_12572, f_12573, f_12574,
                            f_12575, f_12576, f_12577, f_12578, f_12579,
                            f_12580, f_12581, f_12582, x_12598)
          let {bool dim_match_12610} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12611} = eq_i64(d_12571, n_12568)
          let {bool match_12612} = logand(dim_match_12611, dim_match_12610)
          let {cert empty_or_match_cert_12613} =
            assert(match_12612, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12614 aliases res_12599
          let {[d_12570][d_12571]f32 result_proper_shape_12614} =
            <empty_or_match_cert_12613>
            reshape((~d_12570, ~d_12571), res_12599)
          let {bool dim_match_12615} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12616} = eq_i64(d_12571, n_12568)
          let {bool match_12617} = logand(dim_match_12616, dim_match_12615)
          let {cert empty_or_match_cert_12618} =
            assert(match_12617, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12619 aliases res_12600
          let {[d_12570][d_12571]f32 result_proper_shape_12619} =
            <empty_or_match_cert_12618>
            reshape((~d_12570, ~d_12571), res_12600)
          let {bool dim_match_12620} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12621} = eq_i64(d_12571, n_12568)
          let {bool match_12622} = logand(dim_match_12621, dim_match_12620)
          let {cert empty_or_match_cert_12623} =
            assert(match_12622, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12624 aliases res_12601
          let {[d_12570][d_12571]f32 result_proper_shape_12624} =
            <empty_or_match_cert_12623>
            reshape((~d_12570, ~d_12571), res_12601)
          let {bool dim_match_12625} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12626} = eq_i64(d_12571, n_12568)
          let {bool match_12627} = logand(dim_match_12626, dim_match_12625)
          let {cert empty_or_match_cert_12628} =
            assert(match_12627, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12629 aliases res_12602
          let {[d_12570][d_12571]i64 result_proper_shape_12629} =
            <empty_or_match_cert_12628>
            reshape((~d_12570, ~d_12571), res_12602)
          let {bool dim_match_12630} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12631} = eq_i64(d_12571, n_12568)
          let {bool match_12632} = logand(dim_match_12631, dim_match_12630)
          let {cert empty_or_match_cert_12633} =
            assert(match_12632, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12634 aliases res_12603
          let {[d_12570][d_12571]f32 result_proper_shape_12634} =
            <empty_or_match_cert_12633>
            reshape((~d_12570, ~d_12571), res_12603)
          let {bool dim_match_12635} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12636} = eq_i64(d_12571, n_12568)
          let {bool match_12637} = logand(dim_match_12636, dim_match_12635)
          let {cert empty_or_match_cert_12638} =
            assert(match_12637, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12639 aliases res_12604
          let {[d_12570][d_12571]f32 result_proper_shape_12639} =
            <empty_or_match_cert_12638>
            reshape((~d_12570, ~d_12571), res_12604)
          let {bool dim_match_12640} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12641} = eq_i64(d_12571, n_12568)
          let {bool match_12642} = logand(dim_match_12641, dim_match_12640)
          let {cert empty_or_match_cert_12643} =
            assert(match_12642, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12644 aliases res_12605
          let {[d_12570][d_12571]f32 result_proper_shape_12644} =
            <empty_or_match_cert_12643>
            reshape((~d_12570, ~d_12571), res_12605)
          let {bool dim_match_12645} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12646} = eq_i64(d_12571, n_12568)
          let {bool match_12647} = logand(dim_match_12646, dim_match_12645)
          let {cert empty_or_match_cert_12648} =
            assert(match_12647, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12649 aliases res_12606
          let {[d_12570][d_12571]f32 result_proper_shape_12649} =
            <empty_or_match_cert_12648>
            reshape((~d_12570, ~d_12571), res_12606)
          let {bool dim_match_12650} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12651} = eq_i64(d_12571, n_12568)
          let {bool match_12652} = logand(dim_match_12651, dim_match_12650)
          let {cert empty_or_match_cert_12653} =
            assert(match_12652, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12654 aliases res_12607
          let {[d_12570][d_12571]f32 result_proper_shape_12654} =
            <empty_or_match_cert_12653>
            reshape((~d_12570, ~d_12571), res_12607)
          let {bool dim_match_12655} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12656} = eq_i64(d_12571, n_12568)
          let {bool match_12657} = logand(dim_match_12656, dim_match_12655)
          let {cert empty_or_match_cert_12658} =
            assert(match_12657, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12659 aliases res_12608
          let {[d_12570][d_12571]f32 result_proper_shape_12659} =
            <empty_or_match_cert_12658>
            reshape((~d_12570, ~d_12571), res_12608)
          let {bool dim_match_12660} = eq_i64(d_12570, steps_12569)
          let {bool dim_match_12661} = eq_i64(d_12571, n_12568)
          let {bool match_12662} = logand(dim_match_12661, dim_match_12660)
          let {cert empty_or_match_cert_12663} =
            assert(match_12662, "not all iterations produce same shape",
                   "unknown location")
          -- result_proper_shape_12664 aliases res_12609
          let {[d_12570][d_12571]f32 result_proper_shape_12664} =
            <empty_or_match_cert_12663>
            reshape((~d_12570, ~d_12571), res_12609)
          in {result_proper_shape_12614, result_proper_shape_12619,
              result_proper_shape_12624, result_proper_shape_12629,
              result_proper_shape_12634, result_proper_shape_12639,
              result_proper_shape_12644, result_proper_shape_12649,
              result_proper_shape_12654, result_proper_shape_12659,
              result_proper_shape_12664},
        as_12583)
  let {i64 ret₂_12665} = n_12566
  in {res_12584, res_12585, res_12586, res_12587, res_12588, res_12589,
      res_12590, res_12591, res_12592, res_12593, res_12594}
}

fun {bool, bool, bool, bool, bool, bool, bool, bool}
lifted_0_expand_outer_reduce_7994 (bool expand_reduce_12666,
                                   bool expand_reduce_12667,
                                   bool expand_reduce_12668,
                                   bool expand_reduce_12669,
                                   bool expand_reduce_12670,
                                   bool expand_reduce_12671,
                                   bool expand_reduce_12672, bool sz_12673) = {
  {expand_reduce_12666, expand_reduce_12667, expand_reduce_12668,
   expand_reduce_12669, expand_reduce_12670, expand_reduce_12671,
   expand_reduce_12672, sz_12673}
}

fun {bool, bool, bool, bool, bool, bool, bool, bool, bool}
lifted_1_expand_outer_reduce_7995 (bool expand_reduce_12674,
                                   bool expand_reduce_12675,
                                   bool expand_reduce_12676,
                                   bool expand_reduce_12677,
                                   bool expand_reduce_12678,
                                   bool expand_reduce_12679,
                                   bool expand_reduce_12680, bool sz_12681,
                                   bool get_12682) = {
  {expand_reduce_12674, expand_reduce_12675, expand_reduce_12676,
   expand_reduce_12677, expand_reduce_12678, expand_reduce_12679,
   expand_reduce_12680, get_12682, sz_12681}
}

fun {bool, bool, bool, bool, bool, bool, bool, bool, bool, bool}
lifted_2_expand_outer_reduce_7998 (bool expand_reduce_12683,
                                   bool expand_reduce_12684,
                                   bool expand_reduce_12685,
                                   bool expand_reduce_12686,
                                   bool expand_reduce_12687,
                                   bool expand_reduce_12688,
                                   bool expand_reduce_12689, bool get_12690,
                                   bool sz_12691, bool op_12692) = {
  {expand_reduce_12683, expand_reduce_12684, expand_reduce_12685,
   expand_reduce_12686, expand_reduce_12687, expand_reduce_12688,
   expand_reduce_12689, get_12690, op_12692, sz_12691}
}

fun {bool, bool, bool, bool, bool, bool, bool, bool, f32, bool, bool}
lifted_3_expand_outer_reduce_7999 (bool expand_reduce_12693,
                                   bool expand_reduce_12694,
                                   bool expand_reduce_12695,
                                   bool expand_reduce_12696,
                                   bool expand_reduce_12697,
                                   bool expand_reduce_12698,
                                   bool expand_reduce_12699, bool get_12700,
                                   bool op_12701, bool sz_12702,
                                   f32 ne_12703) = {
  {expand_reduce_12693, expand_reduce_12694, expand_reduce_12695,
   expand_reduce_12696, expand_reduce_12697, expand_reduce_12698,
   expand_reduce_12699, get_12700, ne_12703, op_12701, sz_12702}
}

fun {bool, bool, bool, bool, bool, bool, bool, bool}
lifted_0_expand_reduce_8000 (bool map2_12704, bool map2_12705, bool map_12706,
                             bool segmented_reduce_12707,
                             bool segmented_reduce_12708,
                             bool segmented_reduce_12709,
                             bool segmented_reduce_12710, bool sz_12711) = {
  {map2_12704, map2_12705, map_12706, segmented_reduce_12707,
   segmented_reduce_12708, segmented_reduce_12709, segmented_reduce_12710,
   sz_12711}
}

fun {bool, f32, bool, bool, bool, bool, bool, bool, bool, bool, bool}
lifted_1_expand_reduce_8001 (bool map2_12712, bool map2_12713, bool map_12714,
                             bool segmented_reduce_12715,
                             bool segmented_reduce_12716,
                             bool segmented_reduce_12717,
                             bool segmented_reduce_12718, bool sz_12719,
                             bool get_12720, f32 get_12721, bool get_12722) = {
  {get_12720, get_12721, get_12722, map2_12712, map2_12713, map_12714,
   segmented_reduce_12715, segmented_reduce_12716, segmented_reduce_12717,
   segmented_reduce_12718, sz_12719}
}

fun {bool, f32, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool}
lifted_2_expand_reduce_8002 (bool get_12723, f32 get_12724, bool get_12725,
                             bool map2_12726, bool map2_12727, bool map_12728,
                             bool segmented_reduce_12729,
                             bool segmented_reduce_12730,
                             bool segmented_reduce_12731,
                             bool segmented_reduce_12732, bool sz_12733,
                             bool op_12734) = {
  {get_12723, get_12724, get_12725, map2_12726, map2_12727, map_12728, op_12734,
   segmented_reduce_12729, segmented_reduce_12730, segmented_reduce_12731,
   segmented_reduce_12732, sz_12733}
}

fun {bool, f32, bool, bool, bool, bool, f32, bool, bool, bool, bool, bool, bool}
lifted_3_expand_reduce_8003 (bool get_12735, f32 get_12736, bool get_12737,
                             bool map2_12738, bool map2_12739, bool map_12740,
                             bool op_12741, bool segmented_reduce_12742,
                             bool segmented_reduce_12743,
                             bool segmented_reduce_12744,
                             bool segmented_reduce_12745, bool sz_12746,
                             f32 ne_12747) = {
  {get_12735, get_12736, get_12737, map2_12738, map2_12739, map_12740, ne_12747,
   op_12741, segmented_reduce_12742, segmented_reduce_12743,
   segmented_reduce_12744, segmented_reduce_12745, sz_12746}
}

fun {bool} lifted_0_map_8004 (bool nameless_12748, bool f_12749) = {
  {f_12749}
}

fun {i64} lifted_0_sz_8006 (bool nameless_12750, f32 r_12751, f32 swap_12752,
                            f32 swap_12753, i64 swap_12754, f32 swap_12755,
                            f32 t_12756, f32 vasicek_12757, f32 vasicek_12758,
                            f32 vasicek_12759, f32 vasicek_12760,
                            f32 vasicek_12761) = {
  let {f32 x_12762} = fdiv32(t_12756, swap_12755)
  let {f32 ceil_arg_12763} = fsub32(x_12762, 1.0f32)
  let {f32 payments_12764} =
    ceil_3349(ceil_arg_12763)
  let {f32 payments_12765} = payments_12764
  let {i64 y_12766} =
    f32_1916(payments_12765)
  let {i64 max_arg_12767} = sub64(swap_12754, y_12766)
  let {i64 res_12768} =
    max_1951(max_arg_12767, 0i64)
  in {res_12768}
}

fun {i64} lifted_0_f_8007 (bool sz_12769, f32 x_12770, f32 x_12771, f32 x_12772,
                           i64 x_12773, f32 x_12774, f32 x_12775, f32 x_12776,
                           f32 x_12777, f32 x_12778, f32 x_12779,
                           f32 x_12780) = {
  let {i64 s_12781} =
    lifted_0_sz_8006(sz_12769, x_12770, x_12771, x_12772, x_12773, x_12774,
                     x_12775, x_12776, x_12777, x_12778, x_12779, x_12780)
  let {i64 s_12782} = s_12781
  let {bool cond_12783} = eq_i64(s_12782, 0i64)
  let {i64 res_12784} =
    -- Branch returns: {i64}
    if cond_12783
    then {1i64} else {s_12782}
  in {res_12784}
}

fun {*[n_12785]i64} lifted_1_map_8008 (i64 n_12785, bool f_12786,
                                       [n_12785]f32 as_12787,
                                       [n_12785]f32 as_12788,
                                       [n_12785]f32 as_12789,
                                       [n_12785]i64 as_12790,
                                       [n_12785]f32 as_12791,
                                       [n_12785]f32 as_12792,
                                       [n_12785]f32 as_12793,
                                       [n_12785]f32 as_12794,
                                       [n_12785]f32 as_12795,
                                       [n_12785]f32 as_12796,
                                       [n_12785]f32 as_12797) = {
  let {[n_12785]i64 res_12798} =
    map(n_12785,
        fn {i64} (f32 x_12799, f32 x_12800, f32 x_12801, i64 x_12802,
                  f32 x_12803, f32 x_12804, f32 x_12805, f32 x_12806,
                  f32 x_12807, f32 x_12808, f32 x_12809) =>
          let {i64 res_12810} =
            lifted_0_f_8007(f_12786, x_12799, x_12800, x_12801, x_12802,
                            x_12803, x_12804, x_12805, x_12806, x_12807,
                            x_12808, x_12809)
          in {res_12810},
        as_12787, as_12788, as_12789, as_12790, as_12791, as_12792, as_12793,
        as_12794, as_12795, as_12796, as_12797)
  let {i64 ret₂_12811} = n_12785
  in {res_12798}
}

fun {bool, bool} lifted_0_map2_8011 (bool map_12812, bool f_12813) = {
  {f_12813, map_12812}
}

fun {[n_12814]i64, bool, bool} lifted_1_map2_8012 (i64 n_12814, bool f_12815,
                                                   bool map_12816,
                                                   [n_12814]i64 as_12817) = {
  {as_12817, f_12815, map_12816}
}

fun {bool} lifted_0_map_8013 (bool nameless_12818, bool f_12819) = {
  {f_12819}
}

fun {i64} lifted_0_f_8015 (bool nameless_12820, i64 x_12821) = {
  {x_12821}
}

fun {bool} lifted_1_f_8016 (i64 x_12822, i64 x_12823) = {
  let {bool res_12824} = eq_i64(x_12822, x_12823)
  let {bool res_12825} = not res_12824
  in {res_12825}
}

fun {bool} lifted_0_f_8017 (bool f_12826, i64 a_12827, i64 b_12828) = {
  let {i64 lifted_1_f_arg_12829} =
    lifted_0_f_8015(f_12826, a_12827)
  let {bool res_12830} =
    lifted_1_f_8016(lifted_1_f_arg_12829, b_12828)
  in {res_12830}
}

fun {*[n_12831]bool} lifted_1_map_8018 (i64 n_12831, bool f_12832,
                                        [n_12831]i64 as_12833,
                                        [n_12831]i64 as_12834) = {
  let {[n_12831]bool res_12835} =
    map(n_12831,
        fn {bool} (i64 x_12836, i64 x_12837) =>
          let {bool res_12838} =
            lifted_0_f_8017(f_12832, x_12836, x_12837)
          in {res_12838},
        as_12833, as_12834)
  let {i64 ret₂_12839} = n_12831
  in {res_12835}
}

fun {*[n_12840]bool} lifted_2_map2_8019 (i64 n_12840, [n_12840]i64 as_12841,
                                         bool f_12842, bool map_12843,
                                         [n_12840]i64 bs_12844) = {
  -- lifted_1_map_arg_12845 aliases as_12841, bs_12844
  -- lifted_1_map_arg_12846 aliases as_12841, bs_12844
  let {[n_12840]i64 lifted_1_map_arg_12845,
       [n_12840]i64 lifted_1_map_arg_12846} =
    zip2_7749(n_12840, as_12841, bs_12844)
  let {bool lifted_1_map_arg_12847} =
    lifted_0_map_8013(map_12843, f_12842)
  let {[n_12840]bool res_12848} =
    lifted_1_map_8018(n_12840, lifted_1_map_arg_12847, lifted_1_map_arg_12845,
                      lifted_1_map_arg_12846)
  in {res_12848}
}

fun {[?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32, bool, f32, bool, bool}
lifted_0_map2_8020 (i64 impl₀_12849, bool map_12850, [impl₀_12849]f32 f_12851,
                    [impl₀_12849]f32 f_12852, [impl₀_12849]f32 f_12853,
                    [impl₀_12849]i64 f_12854, [impl₀_12849]f32 f_12855,
                    [impl₀_12849]f32 f_12856, [impl₀_12849]f32 f_12857,
                    [impl₀_12849]f32 f_12858, [impl₀_12849]f32 f_12859,
                    [impl₀_12849]f32 f_12860, [impl₀_12849]f32 f_12861,
                    bool f_12862, f32 f_12863, bool f_12864) = {
  {impl₀_12849, f_12851, f_12852, f_12853, f_12854, f_12855, f_12856, f_12857,
   f_12858, f_12859, f_12860, f_12861, f_12862, f_12863, f_12864, map_12850}
}

fun {[n_12865]i64, [?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32, [?0]f32, [?0]f32, bool, f32, bool, bool}
lifted_1_map2_8021 (i64 n_12865, i64 impl₀_12866, [impl₀_12866]f32 f_12867,
                    [impl₀_12866]f32 f_12868, [impl₀_12866]f32 f_12869,
                    [impl₀_12866]i64 f_12870, [impl₀_12866]f32 f_12871,
                    [impl₀_12866]f32 f_12872, [impl₀_12866]f32 f_12873,
                    [impl₀_12866]f32 f_12874, [impl₀_12866]f32 f_12875,
                    [impl₀_12866]f32 f_12876, [impl₀_12866]f32 f_12877,
                    bool f_12878, f32 f_12879, bool f_12880, bool map_12881,
                    [n_12865]i64 as_12882) = {
  {impl₀_12866, as_12882, f_12867, f_12868, f_12869, f_12870, f_12871, f_12872,
   f_12873, f_12874, f_12875, f_12876, f_12877, f_12878, f_12879, f_12880,
   map_12881}
}

fun {[?0]f32, [?0]f32, [?0]f32, [?0]i64, [?0]f32, [?0]f32, [?0]f32, [?0]f32,
     [?0]f32, [?0]f32, [?0]f32, bool, f32, bool}
lifted_0_map_8022 (i64 impl₀_12883, bool nameless_12884,
                   [impl₀_12883]f32 f_12885, [impl₀_12883]f32 f_12886,
                   [impl₀_12883]f32 f_12887, [impl₀_12883]i64 f_12888,
                   [impl₀_12883]f32 f_12889, [impl₀_12883]f32 f_12890,
                   [impl₀_12883]f32 f_12891, [impl₀_12883]f32 f_12892,
                   [impl₀_12883]f32 f_12893, [impl₀_12883]f32 f_12894,
                   [impl₀_12883]f32 f_12895, bool f_12896, f32 f_12897,
                   bool f_12898) = {
  {impl₀_12883, f_12885, f_12886, f_12887, f_12888, f_12889, f_12890, f_12891,
   f_12892, f_12893, f_12894, f_12895, f_12896, f_12897, f_12898}
}

fun {[impl₀_12899]f32, [impl₀_12899]f32, [impl₀_12899]f32, [impl₀_12899]i64,
     [impl₀_12899]f32, [impl₀_12899]f32, [impl₀_12899]f32, [impl₀_12899]f32,
     [impl₀_12899]f32, [impl₀_12899]f32, [impl₀_12899]f32, bool, f32, bool, i64}
lifted_0_f_8024 (i64 impl₀_12899, [impl₀_12899]f32 arr_12900,
                 [impl₀_12899]f32 arr_12901, [impl₀_12899]f32 arr_12902,
                 [impl₀_12899]i64 arr_12903, [impl₀_12899]f32 arr_12904,
                 [impl₀_12899]f32 arr_12905, [impl₀_12899]f32 arr_12906,
                 [impl₀_12899]f32 arr_12907, [impl₀_12899]f32 arr_12908,
                 [impl₀_12899]f32 arr_12909, [impl₀_12899]f32 arr_12910,
                 bool get_12911, f32 get_12912, bool get_12913, i64 i_12914) = {
  {arr_12900, arr_12901, arr_12902, arr_12903, arr_12904, arr_12905, arr_12906,
   arr_12907, arr_12908, arr_12909, arr_12910, get_12911, get_12912, get_12913,
   i_12914}
}

fun {bool, f32, bool, f32, f32, f32, i64, f32, f32, f32, f32, f32, f32, f32}
lifted_0_get_8025 (bool get_12915, f32 ne_12916, bool sz_12917, f32 x_12918,
                   f32 x_12919, f32 x_12920, i64 x_12921, f32 x_12922,
                   f32 x_12923, f32 x_12924, f32 x_12925, f32 x_12926,
                   f32 x_12927, f32 x_12928) = {
  {get_12915, ne_12916, sz_12917, x_12918, x_12919, x_12920, x_12921, x_12922,
   x_12923, x_12924, x_12925, x_12926, x_12927, x_12928}
}

fun {i64} lifted_0_sz_8026 (bool nameless_12929, f32 r_12930, f32 swap_12931,
                            f32 swap_12932, i64 swap_12933, f32 swap_12934,
                            f32 t_12935, f32 vasicek_12936, f32 vasicek_12937,
                            f32 vasicek_12938, f32 vasicek_12939,
                            f32 vasicek_12940) = {
  let {f32 x_12941} = fdiv32(t_12935, swap_12934)
  let {f32 ceil_arg_12942} = fsub32(x_12941, 1.0f32)
  let {f32 payments_12943} =
    ceil_3349(ceil_arg_12942)
  let {f32 payments_12944} = payments_12943
  let {i64 y_12945} =
    f32_1916(payments_12944)
  let {i64 max_arg_12946} = sub64(swap_12933, y_12945)
  let {i64 res_12947} =
    max_1951(max_arg_12946, 0i64)
  in {res_12947}
}

fun {f32, f32, f32, i64, f32, f32, f32, f32, f32, f32, f32}
lifted_0_get_8027 (bool nameless_12948, f32 r_12949, f32 swap_12950,
                   f32 swap_12951, i64 swap_12952, f32 swap_12953, f32 t_12954,
                   f32 vasicek_12955, f32 vasicek_12956, f32 vasicek_12957,
                   f32 vasicek_12958, f32 vasicek_12959) = {
  {r_12949, swap_12950, swap_12951, swap_12952, swap_12953, t_12954,
   vasicek_12955, vasicek_12956, vasicek_12957, vasicek_12958, vasicek_12959}
}

fun {f32} lifted_1_get_8028 (f32 r_12960, f32 swap_12961, f32 swap_12962,
                             i64 swap_12963, f32 swap_12964, f32 t_12965,
                             f32 vasicek_12966, f32 vasicek_12967,
                             f32 vasicek_12968, f32 vasicek_12969,
                             f32 vasicek_12970, i64 i_12971) = {
  let {f32 ceil_arg_12972} = fdiv32(t_12965, swap_12964)
  let {f32 x_12973} =
    ceil_3349(ceil_arg_12972)
  let {f32 start_12974} = fmul32(x_12973, swap_12964)
  let {f32 start_12975} = start_12974
  let {f32 coef_12976} =
    coef_get_7327(r_12960, swap_12961, swap_12962, swap_12963, swap_12964,
                  t_12965, vasicek_12966, vasicek_12967, vasicek_12968,
                  vasicek_12969, vasicek_12970, i_12971)
  let {f32 coef_12977} = coef_12976
  let {f32 x_12978} =
    i64_3244(i_12971)
  let {f32 y_12979} = fmul32(x_12978, swap_12964)
  let {f32 bondprice_arg_12980} = fadd32(start_12975, y_12979)
  let {f32 price_12981} =
    bondprice_7156(vasicek_12966, vasicek_12967, vasicek_12968, vasicek_12969,
                   vasicek_12970, r_12960, t_12965, bondprice_arg_12980)
  let {f32 price_12982} = price_12981
  let {f32 res_12983} = fmul32(coef_12977, price_12982)
  in {res_12983}
}

fun {f32} lifted_1_get_8029 (bool get_12984, f32 ne_12985, bool sz_12986,
                             f32 x_12987, f32 x_12988, f32 x_12989, i64 x_12990,
                             f32 x_12991, f32 x_12992, f32 x_12993, f32 x_12994,
                             f32 x_12995, f32 x_12996, f32 x_12997,
                             i64 i_12998) = {
  let {i64 x_12999} =
    lifted_0_sz_8026(sz_12986, x_12987, x_12988, x_12989, x_12990, x_12991,
                     x_12992, x_12993, x_12994, x_12995, x_12996, x_12997)
  let {bool cond_13000} = eq_i64(x_12999, 0i64)
  let {f32 res_13001} =
    -- Branch returns: {f32}
    if cond_13000
    then {ne_12985} else {
      let {f32 lifted_1_get_arg_13002, f32 lifted_1_get_arg_13003,
           f32 lifted_1_get_arg_13004, i64 lifted_1_get_arg_13005,
           f32 lifted_1_get_arg_13006, f32 lifted_1_get_arg_13007,
           f32 lifted_1_get_arg_13008, f32 lifted_1_get_arg_13009,
           f32 lifted_1_get_arg_13010, f32 lifted_1_get_arg_13011,
           f32 lifted_1_get_arg_13012} =
        lifted_0_get_8027(get_12984, x_12987, x_12988, x_12989, x_12990,
                          x_12991, x_12992, x_12993, x_12994, x_12995, x_12996,
                          x_12997)
      let {f32 res_13013} =
        lifted_1_get_8028(lifted_1_get_arg_13002, lifted_1_get_arg_13003,
                          lifted_1_get_arg_13004, lifted_1_get_arg_13005,
                          lifted_1_get_arg_13006, lifted_1_get_arg_13007,
                          lifted_1_get_arg_13008, lifted_1_get_arg_13009,
                          lifted_1_get_arg_13010, lifted_1_get_arg_13011,
                          lifted_1_get_arg_13012, i_12998)
      in {res_13013}
    }
  in {res_13001}
}

fun {f32} lifted_1_f_8030 (i64 impl₀_13014, [impl₀_13014]f32 arr_13015,
                           [impl₀_13014]f32 arr_13016,
                           [impl₀_13014]f32 arr_13017,
                           [impl₀_13014]i64 arr_13018,
                           [impl₀_13014]f32 arr_13019,
                           [impl₀_13014]f32 arr_13020,
                           [impl₀_13014]f32 arr_13021,
                           [impl₀_13014]f32 arr_13022,
                           [impl₀_13014]f32 arr_13023,
                           [impl₀_13014]f32 arr_13024,
                           [impl₀_13014]f32 arr_13025, bool get_13026,
                           f32 get_13027, bool get_13028, i64 i_13029,
                           i64 j_13030) = {
  let {bool x_13031} = sle64(0i64, i_13029)
  let {bool y_13032} = slt64(i_13029, impl₀_13014)
  let {bool bounds_check_13033} = logand(x_13031, y_13032)
  let {cert index_certs_13034} =
    assert(bounds_check_13033, "Index [", i_13029,
                               "] out of bounds for array of shape [",
                               impl₀_13014, "].",
           "lib/github.com/diku-dk/segmented/segmented.fut:90:30-35")
  let {f32 lifted_0_get_arg_13035} =
    <index_certs_13034>
    arr_13015[i_13029]
  let {f32 lifted_0_get_arg_13036} =
    <index_certs_13034>
    arr_13016[i_13029]
  let {f32 lifted_0_get_arg_13037} =
    <index_certs_13034>
    arr_13017[i_13029]
  let {i64 lifted_0_get_arg_13038} =
    <index_certs_13034>
    arr_13018[i_13029]
  let {f32 lifted_0_get_arg_13039} =
    <index_certs_13034>
    arr_13019[i_13029]
  let {f32 lifted_0_get_arg_13040} =
    <index_certs_13034>
    arr_13020[i_13029]
  let {f32 lifted_0_get_arg_13041} =
    <index_certs_13034>
    arr_13021[i_13029]
  let {f32 lifted_0_get_arg_13042} =
    <index_certs_13034>
    arr_13022[i_13029]
  let {f32 lifted_0_get_arg_13043} =
    <index_certs_13034>
    arr_13023[i_13029]
  let {f32 lifted_0_get_arg_13044} =
    <index_certs_13034>
    arr_13024[i_13029]
  let {f32 lifted_0_get_arg_13045} =
    <index_certs_13034>
    arr_13025[i_13029]
  let {bool lifted_1_get_arg_13046, f32 lifted_1_get_arg_13047,
       bool lifted_1_get_arg_13048, f32 lifted_1_get_arg_13049,
       f32 lifted_1_get_arg_13050, f32 lifted_1_get_arg_13051,
       i64 lifted_1_get_arg_13052, f32 lifted_1_get_arg_13053,
       f32 lifted_1_get_arg_13054, f32 lifted_1_get_arg_13055,
       f32 lifted_1_get_arg_13056, f32 lifted_1_get_arg_13057,
       f32 lifted_1_get_arg_13058, f32 lifted_1_get_arg_13059} =
    lifted_0_get_8025(get_13026, get_13027, get_13028, lifted_0_get_arg_13035,
                      lifted_0_get_arg_13036, lifted_0_get_arg_13037,
                      lifted_0_get_arg_13038, lifted_0_get_arg_13039,
                      lifted_0_get_arg_13040, lifted_0_get_arg_13041,
                      lifted_0_get_arg_13042, lifted_0_get_arg_13043,
                      lifted_0_get_arg_13044, lifted_0_get_arg_13045)
  let {f32 res_13060} =
    lifted_1_get_8029(lifted_1_get_arg_13046, lifted_1_get_arg_13047,
                      lifted_1_get_arg_13048, lifted_1_get_arg_13049,
                      lifted_1_get_arg_13050, lifted_1_get_arg_13051,
                      lifted_1_get_arg_13052, lifted_1_get_arg_13053,
                      lifted_1_get_arg_13054, lifted_1_get_arg_13055,
                      lifted_1_get_arg_13056, lifted_1_get_arg_13057,
                      lifted_1_get_arg_13058, lifted_1_get_arg_13059, j_13030)
  in {res_13060}
}

fun {f32} lifted_0_f_8031 (i64 impl₀_13061, [impl₀_13061]f32 f_13062,
                           [impl₀_13061]f32 f_13063, [impl₀_13061]f32 f_13064,
                           [impl₀_13061]i64 f_13065, [impl₀_13061]f32 f_13066,
                           [impl₀_13061]f32 f_13067, [impl₀_13061]f32 f_13068,
                           [impl₀_13061]f32 f_13069, [impl₀_13061]f32 f_13070,
                           [impl₀_13061]f32 f_13071, [impl₀_13061]f32 f_13072,
                           bool f_13073, f32 f_13074, bool f_13075, i64 a_13076,
                           i64 b_13077) = {
  -- lifted_1_f_arg_13078 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13079 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13080 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13081 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13082 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13083 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13084 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13085 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13086 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13087 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  -- lifted_1_f_arg_13088 aliases f_13062, f_13063, f_13064, f_13065, f_13066, f_13067, f_13068, f_13069, f_13070, f_13071, f_13072
  let {[impl₀_13061]f32 lifted_1_f_arg_13078,
       [impl₀_13061]f32 lifted_1_f_arg_13079,
       [impl₀_13061]f32 lifted_1_f_arg_13080,
       [impl₀_13061]i64 lifted_1_f_arg_13081,
       [impl₀_13061]f32 lifted_1_f_arg_13082,
       [impl₀_13061]f32 lifted_1_f_arg_13083,
       [impl₀_13061]f32 lifted_1_f_arg_13084,
       [impl₀_13061]f32 lifted_1_f_arg_13085,
       [impl₀_13061]f32 lifted_1_f_arg_13086,
       [impl₀_13061]f32 lifted_1_f_arg_13087,
       [impl₀_13061]f32 lifted_1_f_arg_13088, bool lifted_1_f_arg_13089,
       f32 lifted_1_f_arg_13090, bool lifted_1_f_arg_13091,
       i64 lifted_1_f_arg_13092} =
    lifted_0_f_8024(impl₀_13061, f_13062, f_13063, f_13064, f_13065, f_13066,
                    f_13067, f_13068, f_13069, f_13070, f_13071, f_13072,
                    f_13073, f_13074, f_13075, a_13076)
  let {f32 res_13093} =
    lifted_1_f_8030(impl₀_13061, lifted_1_f_arg_13078, lifted_1_f_arg_13079,
                    lifted_1_f_arg_13080, lifted_1_f_arg_13081,
                    lifted_1_f_arg_13082, lifted_1_f_arg_13083,
                    lifted_1_f_arg_13084, lifted_1_f_arg_13085,
                    lifted_1_f_arg_13086, lifted_1_f_arg_13087,
                    lifted_1_f_arg_13088, lifted_1_f_arg_13089,
                    lifted_1_f_arg_13090, lifted_1_f_arg_13091,
                    lifted_1_f_arg_13092, b_13077)
  in {res_13093}
}

fun {*[n_13094]f32} lifted_1_map_8032 (i64 n_13094, i64 impl₀_13095,
                                       [impl₀_13095]f32 f_13096,
                                       [impl₀_13095]f32 f_13097,
                                       [impl₀_13095]f32 f_13098,
                                       [impl₀_13095]i64 f_13099,
                                       [impl₀_13095]f32 f_13100,
                                       [impl₀_13095]f32 f_13101,
                                       [impl₀_13095]f32 f_13102,
                                       [impl₀_13095]f32 f_13103,
                                       [impl₀_13095]f32 f_13104,
                                       [impl₀_13095]f32 f_13105,
                                       [impl₀_13095]f32 f_13106, bool f_13107,
                                       f32 f_13108, bool f_13109,
                                       [n_13094]i64 as_13110,
                                       [n_13094]i64 as_13111) = {
  let {[n_13094]f32 res_13112} =
    map(n_13094,
        fn {f32} (i64 x_13113, i64 x_13114) =>
          let {f32 res_13115} =
            lifted_0_f_8031(impl₀_13095, f_13096, f_13097, f_13098, f_13099,
                            f_13100, f_13101, f_13102, f_13103, f_13104,
                            f_13105, f_13106, f_13107, f_13108, f_13109,
                            x_13113, x_13114)
          in {res_13115},
        as_13110, as_13111)
  let {i64 ret₂_13116} = n_13094
  in {res_13112}
}

fun {*[n_13117]f32} lifted_2_map2_8033 (i64 n_13117, i64 impl₀_13118,
                                        [n_13117]i64 as_13119,
                                        [impl₀_13118]f32 f_13120,
                                        [impl₀_13118]f32 f_13121,
                                        [impl₀_13118]f32 f_13122,
                                        [impl₀_13118]i64 f_13123,
                                        [impl₀_13118]f32 f_13124,
                                        [impl₀_13118]f32 f_13125,
                                        [impl₀_13118]f32 f_13126,
                                        [impl₀_13118]f32 f_13127,
                                        [impl₀_13118]f32 f_13128,
                                        [impl₀_13118]f32 f_13129,
                                        [impl₀_13118]f32 f_13130, bool f_13131,
                                        f32 f_13132, bool f_13133,
                                        bool map_13134,
                                        [n_13117]i64 bs_13135) = {
  -- lifted_1_map_arg_13136 aliases as_13119, bs_13135
  -- lifted_1_map_arg_13137 aliases as_13119, bs_13135
  let {[n_13117]i64 lifted_1_map_arg_13136,
       [n_13117]i64 lifted_1_map_arg_13137} =
    zip2_7749(n_13117, as_13119, bs_13135)
  -- lifted_1_map_arg_13139 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13140 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13141 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13142 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13143 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13144 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13145 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13146 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13147 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13148 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  -- lifted_1_map_arg_13149 aliases f_13120, f_13121, f_13122, f_13123, f_13124, f_13125, f_13126, f_13127, f_13128, f_13129, f_13130
  let {i64 size_13138;
       [size_13138]f32 lifted_1_map_arg_13139,
       [size_13138]f32 lifted_1_map_arg_13140,
       [size_13138]f32 lifted_1_map_arg_13141,
       [size_13138]i64 lifted_1_map_arg_13142,
       [size_13138]f32 lifted_1_map_arg_13143,
       [size_13138]f32 lifted_1_map_arg_13144,
       [size_13138]f32 lifted_1_map_arg_13145,
       [size_13138]f32 lifted_1_map_arg_13146,
       [size_13138]f32 lifted_1_map_arg_13147,
       [size_13138]f32 lifted_1_map_arg_13148,
       [size_13138]f32 lifted_1_map_arg_13149, bool lifted_1_map_arg_13150,
       f32 lifted_1_map_arg_13151, bool lifted_1_map_arg_13152} =
    lifted_0_map_8022(impl₀_13118, map_13134, f_13120, f_13121, f_13122,
                      f_13123, f_13124, f_13125, f_13126, f_13127, f_13128,
                      f_13129, f_13130, f_13131, f_13132, f_13133)
  let {[n_13117]f32 res_13153} =
    lifted_1_map_8032(n_13117, size_13138, lifted_1_map_arg_13139,
                      lifted_1_map_arg_13140, lifted_1_map_arg_13141,
                      lifted_1_map_arg_13142, lifted_1_map_arg_13143,
                      lifted_1_map_arg_13144, lifted_1_map_arg_13145,
                      lifted_1_map_arg_13146, lifted_1_map_arg_13147,
                      lifted_1_map_arg_13148, lifted_1_map_arg_13149,
                      lifted_1_map_arg_13150, lifted_1_map_arg_13151,
                      lifted_1_map_arg_13152, lifted_1_map_arg_13136,
                      lifted_1_map_arg_13137)
  in {res_13153}
}

fun {bool, bool, bool, bool, bool}
lifted_0_segmented_reduce_8034 (bool map2_13154, bool map_13155,
                                bool scan_13156, bool segmented_scan_13157,
                                bool op_13158) = {
  {map2_13154, map_13155, op_13158, scan_13156, segmented_scan_13157}
}

fun {bool, bool, f32, bool, bool, bool}
lifted_1_segmented_reduce_8035 (bool map2_13159, bool map_13160, bool op_13161,
                                bool scan_13162, bool segmented_scan_13163,
                                f32 ne_13164) = {
  {map2_13159, map_13160, ne_13164, op_13161, scan_13162, segmented_scan_13163}
}

fun {[n_13165]bool, bool, bool, f32, bool, bool, bool}
lifted_2_segmented_reduce_8036 (i64 n_13165, bool map2_13166, bool map_13167,
                                f32 ne_13168, bool op_13169, bool scan_13170,
                                bool segmented_scan_13171,
                                [n_13165]bool flags_13172) = {
  {flags_13172, map2_13166, map_13167, ne_13168, op_13169, scan_13170,
   segmented_scan_13171}
}

fun {bool, bool} lifted_0_segmented_scan_8037 (bool scan_13173,
                                               bool op_13174) = {
  {op_13174, scan_13173}
}

fun {f32, bool, bool} lifted_1_segmented_scan_8038 (bool op_13175,
                                                    bool scan_13176,
                                                    f32 ne_13177) = {
  {ne_13177, op_13175, scan_13176}
}

fun {[n_13178]bool, f32, bool, bool} lifted_2_segmented_scan_8039 (i64 n_13178,
                                                                   f32 ne_13179,
                                                                   bool op_13180,
                                                                   bool scan_13181,
                                                                   [n_13178]bool flags_13182) = {
  {flags_13182, ne_13179, op_13180, scan_13181}
}

fun {bool} lifted_0_scan_8040 (bool nameless_13183, bool op_13184) = {
  {op_13184}
}

fun {bool, f32, bool} lifted_1_scan_8041 (bool op_13185, bool 0_13186,
                                          f32 1_13187) = {
  {0_13186, 1_13187, op_13185}
}

fun {bool, f32, bool} lifted_0_op_8044 (bool op_13188, bool x_flag_13189,
                                        f32 x_13190) = {
  {op_13188, x_13190, x_flag_13189}
}

fun {f32} lifted_0_op_8045 (bool nameless_13191, f32 x_13192) = {
  {x_13192}
}

fun {f32} lifted_1_op_8046 (f32 x_13193, f32 x_13194) = {
  let {f32 res_13195} = fadd32(x_13193, x_13194)
  in {res_13195}
}

fun {bool, f32} lifted_1_op_8047 (bool op_13196, f32 x_13197, bool x_flag_13198,
                                  bool y_flag_13199, f32 y_13200) = {
  let {bool res_13201} =
    -- Branch returns: {bool}
    if x_flag_13198
    then {true} else {y_flag_13199}
  let {f32 res_13202} =
    -- Branch returns: {f32}
    if y_flag_13199
    then {y_13200} else {
      let {f32 lifted_1_op_arg_13203} =
        lifted_0_op_8045(op_13196, x_13197)
      let {f32 res_13204} =
        lifted_1_op_8046(lifted_1_op_arg_13203, y_13200)
      in {res_13204}
    }
  in {res_13201, res_13202}
}

fun {*[n_13205]bool, *[n_13205]f32} lifted_2_scan_8048 (i64 n_13205,
                                                        bool 0_13206,
                                                        f32 1_13207,
                                                        bool op_13208,
                                                        [n_13205]bool as_13209,
                                                        [n_13205]f32 as_13210) = {
  let {[n_13205]bool res_13211, [n_13205]f32 res_13212} =
    scanomap(n_13205,
             {fn {bool, f32} (bool x_13213, f32 x_13214, bool x_13215,
                              f32 x_13216) =>
                let {bool lifted_1_op_arg_13217, f32 lifted_1_op_arg_13218,
                     bool lifted_1_op_arg_13219} =
                  lifted_0_op_8044(op_13208, x_13213, x_13214)
                let {bool res_13220, f32 res_13221} =
                  lifted_1_op_8047(lifted_1_op_arg_13217, lifted_1_op_arg_13218,
                                   lifted_1_op_arg_13219, x_13215, x_13216)
                in {res_13220, res_13221},
              {0_13206, 1_13207}},
             fn {bool, f32} (bool x_13222, f32 x_13223) =>
               {x_13222, x_13223},
             as_13209, as_13210)
  let {i64 ret₁_13224} = n_13205
  in {res_13211, res_13212}
}

fun {[n_13225]f32} lifted_3_segmented_scan_8049 (i64 n_13225,
                                                 [n_13225]bool flags_13226,
                                                 f32 ne_13227, bool op_13228,
                                                 bool scan_13229,
                                                 [n_13225]f32 as_13230) = {
  -- lifted_2_scan_arg_13231 aliases flags_13226, as_13230
  -- lifted_2_scan_arg_13232 aliases flags_13226, as_13230
  let {[n_13225]bool lifted_2_scan_arg_13231,
       [n_13225]f32 lifted_2_scan_arg_13232} =
    zip_7775(n_13225, flags_13226, as_13230)
  let {bool lifted_1_scan_arg_13233} =
    lifted_0_scan_8040(scan_13229, op_13228)
  let {bool lifted_2_scan_arg_13234, f32 lifted_2_scan_arg_13235,
       bool lifted_2_scan_arg_13236} =
    lifted_1_scan_8041(lifted_1_scan_arg_13233, false, ne_13227)
  let {[n_13225]bool unzip_arg_13237, [n_13225]f32 unzip_arg_13238} =
    lifted_2_scan_8048(n_13225, lifted_2_scan_arg_13234,
                       lifted_2_scan_arg_13235, lifted_2_scan_arg_13236,
                       lifted_2_scan_arg_13231, lifted_2_scan_arg_13232)
  -- res_13239 aliases unzip_arg_13237, unzip_arg_13238
  -- res_13240 aliases unzip_arg_13237, unzip_arg_13238
  let {[n_13225]bool res_13239, [n_13225]f32 res_13240} =
    unzip_7771(n_13225, unzip_arg_13237, unzip_arg_13238)
  in {res_13240}
}

fun {bool} lifted_0_map_8050 (bool nameless_13241, bool f_13242) = {
  {f_13242}
}

fun {i64} lifted_0_f_8052 (bool nameless_13243, bool x_13244) = {
  let {i64 res_13245} = btoi bool x_13244 to i64
  in {res_13245}
}

fun {*[n_13246]i64} lifted_0_f_8053 (i64 n_13246, bool f_13247,
                                     [n_13246]bool as_13248) = {
  let {[n_13246]i64 res_13249} =
    map(n_13246,
        fn {i64} (bool x_13250) =>
          let {i64 res_13251} =
            lifted_0_f_8052(f_13247, x_13250)
          in {res_13251},
        as_13248)
  let {i64 ret₂_13252} = n_13246
  in {res_13249}
}

fun {[d_13254]i64} lifted_2_|>_8054 (i64 d_13253, i64 d_13254,
                                     [d_13253]bool x_13255, bool f_13256) = {
  let {[d_13253]i64 res_13257} =
    lifted_0_f_8053(d_13253, f_13256, x_13255)
  let {bool dim_match_13258} = eq_i64(d_13254, d_13253)
  let {cert empty_or_match_cert_13259} =
    assert(dim_match_13258,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_13260 aliases res_13257
  let {[d_13254]i64 result_proper_shape_13260} =
    <empty_or_match_cert_13259>
    reshape((~d_13254), res_13257)
  in {result_proper_shape_13260}
}

fun {bool} lifted_0_scan_8057 (bool nameless_13261, bool op_13262) = {
  {op_13262}
}

fun {i64, bool} lifted_1_scan_8058 (bool op_13263, i64 ne_13264) = {
  {ne_13264, op_13263}
}

fun {i64} lifted_0_op_8061 (bool nameless_13265, i64 x_13266) = {
  {x_13266}
}

fun {i64} lifted_1_op_8062 (i64 x_13267, i64 x_13268) = {
  let {i64 res_13269} = add64(x_13267, x_13268)
  in {res_13269}
}

fun {*[n_13270]i64} lifted_0_f_8063 (i64 n_13270, i64 ne_13271, bool op_13272,
                                     [n_13270]i64 as_13273) = {
  let {[n_13270]i64 res_13274} =
    scanomap(n_13270,
             {fn {i64} (i64 x_13275, i64 x_13276) =>
                let {i64 lifted_1_op_arg_13277} =
                  lifted_0_op_8061(op_13272, x_13275)
                let {i64 res_13278} =
                  lifted_1_op_8062(lifted_1_op_arg_13277, x_13276)
                in {res_13278},
              {ne_13271}},
             fn {i64} (i64 x_13279) =>
               {x_13279},
             as_13273)
  let {i64 ret₁_13280} = n_13270
  in {res_13274}
}

fun {[d_13282]i64} lifted_2_|>_8064 (i64 d_13281, i64 d_13282,
                                     [d_13281]i64 x_13283, i64 f_13284,
                                     bool f_13285) = {
  let {[d_13281]i64 res_13286} =
    lifted_0_f_8063(d_13281, f_13284, f_13285, x_13283)
  let {bool dim_match_13287} = eq_i64(d_13282, d_13281)
  let {cert empty_or_match_cert_13288} =
    assert(dim_match_13287,
           "Function return value does not match shape of declared return type.",
           "unknown location")
  -- result_proper_shape_13289 aliases res_13286
  let {[d_13282]i64 result_proper_shape_13289} =
    <empty_or_match_cert_13288>
    reshape((~d_13282), res_13286)
  in {result_proper_shape_13289}
}

fun {bool, bool} lifted_0_map2_8065 (bool map_13290, bool f_13291) = {
  {f_13291, map_13290}
}

fun {[n_13292]i64, bool, bool} lifted_1_map2_8066 (i64 n_13292, bool f_13293,
                                                   bool map_13294,
                                                   [n_13292]i64 as_13295) = {
  {as_13295, f_13293, map_13294}
}

fun {bool} lifted_0_map_8067 (bool nameless_13296, bool f_13297) = {
  {f_13297}
}

fun {i64} lifted_0_f_8069 (bool nameless_13298, i64 i_13299) = {
  {i_13299}
}

fun {i64} lifted_1_f_8070 (i64 i_13300, bool f_13301) = {
  let {i64 res_13302} =
    -- Branch returns: {i64}
    if f_13301
    then {
      let {i64 res_13303} = sub64(i_13300, 1i64)
      in {res_13303}
    } else {
      let {i64 res_13304} = sub64(0i64, 1i64)
      in {res_13304}
    }
  in {res_13302}
}

fun {i64} lifted_0_f_8071 (bool f_13305, i64 a_13306, bool b_13307) = {
  let {i64 lifted_1_f_arg_13308} =
    lifted_0_f_8069(f_13305, a_13306)
  let {i64 res_13309} =
    lifted_1_f_8070(lifted_1_f_arg_13308, b_13307)
  in {res_13309}
}

fun {*[n_13310]i64} lifted_1_map_8072 (i64 n_13310, bool f_13311,
                                       [n_13310]i64 as_13312,
                                       [n_13310]bool as_13313) = {
  let {[n_13310]i64 res_13314} =
    map(n_13310,
        fn {i64} (i64 x_13315, bool x_13316) =>
          let {i64 res_13317} =
            lifted_0_f_8071(f_13311, x_13315, x_13316)
          in {res_13317},
        as_13312, as_13313)
  let {i64 ret₂_13318} = n_13310
  in {res_13314}
}

fun {*[n_13319]i64} lifted_2_map2_8073 (i64 n_13319, [n_13319]i64 as_13320,
                                        bool f_13321, bool map_13322,
                                        [n_13319]bool bs_13323) = {
  -- lifted_1_map_arg_13324 aliases as_13320, bs_13323
  -- lifted_1_map_arg_13325 aliases as_13320, bs_13323
  let {[n_13319]i64 lifted_1_map_arg_13324,
       [n_13319]bool lifted_1_map_arg_13325} =
    zip2_7793(n_13319, as_13320, bs_13323)
  let {bool lifted_1_map_arg_13326} =
    lifted_0_map_8067(map_13322, f_13321)
  let {[n_13319]i64 res_13327} =
    lifted_1_map_8072(n_13319, lifted_1_map_arg_13326, lifted_1_map_arg_13324,
                      lifted_1_map_arg_13325)
  in {res_13327}
}

fun {*[?0]f32} lifted_3_segmented_reduce_8074 (i64 n_13328,
                                               [n_13328]bool flags_13329,
                                               bool map2_13330, bool map_13331,
                                               f32 ne_13332, bool op_13333,
                                               bool scan_13334,
                                               bool segmented_scan_13335,
                                               [n_13328]f32 as_13336) = {
  let {bool lifted_1_segmented_scan_arg_13337,
       bool lifted_1_segmented_scan_arg_13338} =
    lifted_0_segmented_scan_8037(segmented_scan_13335, op_13333)
  let {f32 lifted_2_segmented_scan_arg_13339,
       bool lifted_2_segmented_scan_arg_13340,
       bool lifted_2_segmented_scan_arg_13341} =
    lifted_1_segmented_scan_8038(lifted_1_segmented_scan_arg_13337,
                                 lifted_1_segmented_scan_arg_13338, ne_13332)
  -- lifted_3_segmented_scan_arg_13342 aliases flags_13329
  let {[n_13328]bool lifted_3_segmented_scan_arg_13342,
       f32 lifted_3_segmented_scan_arg_13343,
       bool lifted_3_segmented_scan_arg_13344,
       bool lifted_3_segmented_scan_arg_13345} =
    lifted_2_segmented_scan_8039(n_13328, lifted_2_segmented_scan_arg_13339,
                                 lifted_2_segmented_scan_arg_13340,
                                 lifted_2_segmented_scan_arg_13341, flags_13329)
  -- as'_13346 aliases as_13336, lifted_3_segmented_scan_arg_13342
  let {[n_13328]f32 as'_13346} =
    lifted_3_segmented_scan_8049(n_13328, lifted_3_segmented_scan_arg_13342,
                                 lifted_3_segmented_scan_arg_13343,
                                 lifted_3_segmented_scan_arg_13344,
                                 lifted_3_segmented_scan_arg_13345, as_13336)
  -- as'_13347 aliases as'_13346
  let {[n_13328]f32 as'_13347} = as'_13346
  -- segment_ends_13348 aliases flags_13329
  let {[n_13328]bool segment_ends_13348} =
    rotate_7777(n_13328, 1i64, flags_13329)
  -- segment_ends_13349 aliases segment_ends_13348
  let {[n_13328]bool segment_ends_13349} = segment_ends_13348
  -- binop_p_13350 aliases segment_ends_13349
  let {[n_13328]bool binop_p_13350} = segment_ends_13349
  let {bool binop_p_13351} =
    lifted_0_map_8050(map_13331, true)
  let {bool binop_p_13352} = binop_p_13351
  -- lifted_2_|>_arg_13355 aliases binop_p_13350
  let {i64 size_13353;
       i64 lifted_2_|>_arg_13354, [size_13353]bool lifted_2_|>_arg_13355} =
    |>_7783(n_13328, n_13328, binop_p_13350)
  -- binop_p_13356 aliases lifted_2_|>_arg_13355
  let {[lifted_2_|>_arg_13354]i64 binop_p_13356} =
    lifted_2_|>_8054(size_13353, lifted_2_|>_arg_13354, lifted_2_|>_arg_13355,
                     binop_p_13352)
  -- binop_p_13357 aliases binop_p_13356
  let {[lifted_2_|>_arg_13354]i64 binop_p_13357} = binop_p_13356
  let {bool lifted_1_scan_arg_13358} =
    lifted_0_scan_8057(scan_13334, true)
  let {i64 binop_p_13359, bool binop_p_13360} =
    lifted_1_scan_8058(lifted_1_scan_arg_13358, 0i64)
  let {i64 binop_p_13361} = binop_p_13359
  let {bool binop_p_13362} = binop_p_13360
  -- lifted_2_|>_arg_13365 aliases binop_p_13357
  let {i64 size_13363;
       i64 lifted_2_|>_arg_13364, [size_13363]i64 lifted_2_|>_arg_13365} =
    |>_7780(lifted_2_|>_arg_13354, n_13328, binop_p_13357)
  -- segment_end_offsets_13366 aliases lifted_2_|>_arg_13365
  let {[lifted_2_|>_arg_13364]i64 segment_end_offsets_13366} =
    lifted_2_|>_8064(size_13363, lifted_2_|>_arg_13364, lifted_2_|>_arg_13365,
                     binop_p_13361, binop_p_13362)
  -- segment_end_offsets_13367 aliases segment_end_offsets_13366
  let {[lifted_2_|>_arg_13364]i64 segment_end_offsets_13367} =
    segment_end_offsets_13366
  let {bool cond_13368} = slt64(0i64, n_13328)
  let {i64 num_segments_13369} =
    -- Branch returns: {i64}
    if cond_13368
    then {
      let {i64 res_13370} =
        last_7789(lifted_2_|>_arg_13364, segment_end_offsets_13367)
      in {res_13370}
    } else {0i64}
  let {i64 num_segments_13371} = num_segments_13369
  let {[num_segments_13371]f32 scratch_13372} =
    replicate_7720(num_segments_13371, ne_13332)
  -- scratch_13373 aliases scratch_13372
  let {[num_segments_13371]f32 scratch_13373} = scratch_13372
  let {bool index_13374} = true
  let {bool lifted_1_map2_arg_13375, bool lifted_1_map2_arg_13376} =
    lifted_0_map2_8065(map2_13330, index_13374)
  -- lifted_2_map2_arg_13377 aliases segment_end_offsets_13367
  let {[lifted_2_|>_arg_13364]i64 lifted_2_map2_arg_13377,
       bool lifted_2_map2_arg_13378, bool lifted_2_map2_arg_13379} =
    lifted_1_map2_8066(lifted_2_|>_arg_13364, lifted_1_map2_arg_13375,
                       lifted_1_map2_arg_13376, segment_end_offsets_13367)
  let {bool dim_match_13380} = eq_i64(lifted_2_|>_arg_13364, n_13328)
  let {cert empty_or_match_cert_13381} =
    assert(dim_match_13380, "function arguments of wrong shape",
           "lib/github.com/diku-dk/segmented/segmented.fut:37:23-65")
  -- segment_ends_13382 aliases segment_ends_13349
  let {[lifted_2_|>_arg_13364]bool segment_ends_13382} =
    <empty_or_match_cert_13381>
    reshape((~lifted_2_|>_arg_13364), segment_ends_13349)
  let {[lifted_2_|>_arg_13364]i64 scatter_arg_13383} =
    lifted_2_map2_8073(lifted_2_|>_arg_13364, lifted_2_map2_arg_13377,
                       lifted_2_map2_arg_13378, lifted_2_map2_arg_13379,
                       segment_ends_13382)
  let {bool dim_match_13384} = eq_i64(lifted_2_|>_arg_13364, n_13328)
  let {cert empty_or_match_cert_13385} =
    assert(dim_match_13384, "function arguments of wrong shape",
           "lib/github.com/diku-dk/segmented/segmented.fut:37:6-70")
  -- as'_13386 aliases as'_13347
  let {[lifted_2_|>_arg_13364]f32 as'_13386} =
    <empty_or_match_cert_13385>
    reshape((~lifted_2_|>_arg_13364), as'_13347)
  let {[num_segments_13371]f32 res_13387} =
    -- Consumes scratch_13373
    scatter_7790(num_segments_13371, lifted_2_|>_arg_13364, *scratch_13373,
                 scatter_arg_13383, as'_13386)
  let {i64 d₃₄_13388} = num_segments_13371
  in {num_segments_13371, res_13387}
}

fun {[?0]f32} lifted_4_expand_reduce_8075 (i64 impl₀_13389, bool get_13390,
                                           f32 get_13391, bool get_13392,
                                           bool map2_13393, bool map2_13394,
                                           bool map_13395, f32 ne_13396,
                                           bool op_13397,
                                           bool segmented_reduce_13398,
                                           bool segmented_reduce_13399,
                                           bool segmented_reduce_13400,
                                           bool segmented_reduce_13401,
                                           bool sz_13402,
                                           [impl₀_13389]f32 arr_13403,
                                           [impl₀_13389]f32 arr_13404,
                                           [impl₀_13389]f32 arr_13405,
                                           [impl₀_13389]i64 arr_13406,
                                           [impl₀_13389]f32 arr_13407,
                                           [impl₀_13389]f32 arr_13408,
                                           [impl₀_13389]f32 arr_13409,
                                           [impl₀_13389]f32 arr_13410,
                                           [impl₀_13389]f32 arr_13411,
                                           [impl₀_13389]f32 arr_13412,
                                           [impl₀_13389]f32 arr_13413) = {
  let {bool lifted_1_map_arg_13414} =
    lifted_0_map_8004(map_13395, sz_13402)
  let {[impl₀_13389]i64 szs_13415} =
    lifted_1_map_8008(impl₀_13389, lifted_1_map_arg_13414, arr_13403, arr_13404,
                      arr_13405, arr_13406, arr_13407, arr_13408, arr_13409,
                      arr_13410, arr_13411, arr_13412, arr_13413)
  -- szs_13416 aliases szs_13415
  let {[impl₀_13389]i64 szs_13416} = szs_13415
  -- idxs_13418 aliases szs_13416
  let {i64 size_13417;
       [size_13417]i64 idxs_13418} =
    replicated_iota_7765(impl₀_13389, szs_13416)
  let {i64 ret₆_13419} = size_13417
  -- idxs_13420 aliases idxs_13418
  let {[size_13417]i64 idxs_13420} = idxs_13418
  let {i64 rotate_arg_13421} =
    negate_1948(1i64)
  let {i64 argdim₁₅_13422} = rotate_arg_13421
  -- lifted_2_map2_arg_13423 aliases idxs_13420
  let {[size_13417]i64 lifted_2_map2_arg_13423} =
    rotate_7751(size_13417, rotate_arg_13421, idxs_13420)
  let {bool lifted_1_map2_arg_13424, bool lifted_1_map2_arg_13425} =
    lifted_0_map2_8011(map2_13393, true)
  -- lifted_2_map2_arg_13426 aliases idxs_13420
  let {[size_13417]i64 lifted_2_map2_arg_13426, bool lifted_2_map2_arg_13427,
       bool lifted_2_map2_arg_13428} =
    lifted_1_map2_8012(size_13417, lifted_1_map2_arg_13424,
                       lifted_1_map2_arg_13425, idxs_13420)
  let {[size_13417]bool flags_13429} =
    lifted_2_map2_8019(size_13417, lifted_2_map2_arg_13426,
                       lifted_2_map2_arg_13427, lifted_2_map2_arg_13428,
                       lifted_2_map2_arg_13423)
  -- flags_13430 aliases flags_13429
  let {[size_13417]bool flags_13430} = flags_13429
  -- iotas_13431 aliases flags_13430
  let {[size_13417]i64 iotas_13431} =
    segmented_iota_7768(size_13417, flags_13430)
  -- iotas_13432 aliases iotas_13431
  let {[size_13417]i64 iotas_13432} = iotas_13431
  -- lifted_1_map2_arg_13434 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13435 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13436 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13437 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13438 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13439 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13440 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13441 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13442 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13443 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  -- lifted_1_map2_arg_13444 aliases arr_13403, arr_13404, arr_13405, arr_13406, arr_13407, arr_13408, arr_13409, arr_13410, arr_13411, arr_13412, arr_13413
  let {i64 size_13433;
       [size_13433]f32 lifted_1_map2_arg_13434,
       [size_13433]f32 lifted_1_map2_arg_13435,
       [size_13433]f32 lifted_1_map2_arg_13436,
       [size_13433]i64 lifted_1_map2_arg_13437,
       [size_13433]f32 lifted_1_map2_arg_13438,
       [size_13433]f32 lifted_1_map2_arg_13439,
       [size_13433]f32 lifted_1_map2_arg_13440,
       [size_13433]f32 lifted_1_map2_arg_13441,
       [size_13433]f32 lifted_1_map2_arg_13442,
       [size_13433]f32 lifted_1_map2_arg_13443,
       [size_13433]f32 lifted_1_map2_arg_13444, bool lifted_1_map2_arg_13445,
       f32 lifted_1_map2_arg_13446, bool lifted_1_map2_arg_13447,
       bool lifted_1_map2_arg_13448} =
    lifted_0_map2_8020(impl₀_13389, map2_13394, arr_13403, arr_13404, arr_13405,
                       arr_13406, arr_13407, arr_13408, arr_13409, arr_13410,
                       arr_13411, arr_13412, arr_13413, get_13390, get_13391,
                       get_13392)
  -- lifted_2_map2_arg_13450 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13451 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13452 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13453 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13454 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13455 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13456 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13457 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13458 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13459 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13460 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  -- lifted_2_map2_arg_13461 aliases idxs_13420, lifted_1_map2_arg_13434, lifted_1_map2_arg_13435, lifted_1_map2_arg_13436, lifted_1_map2_arg_13437, lifted_1_map2_arg_13438, lifted_1_map2_arg_13439, lifted_1_map2_arg_13440, lifted_1_map2_arg_13441, lifted_1_map2_arg_13442, lifted_1_map2_arg_13443, lifted_1_map2_arg_13444
  let {i64 size_13449;
       [size_13417]i64 lifted_2_map2_arg_13450,
       [size_13449]f32 lifted_2_map2_arg_13451,
       [size_13449]f32 lifted_2_map2_arg_13452,
       [size_13449]f32 lifted_2_map2_arg_13453,
       [size_13449]i64 lifted_2_map2_arg_13454,
       [size_13449]f32 lifted_2_map2_arg_13455,
       [size_13449]f32 lifted_2_map2_arg_13456,
       [size_13449]f32 lifted_2_map2_arg_13457,
       [size_13449]f32 lifted_2_map2_arg_13458,
       [size_13449]f32 lifted_2_map2_arg_13459,
       [size_13449]f32 lifted_2_map2_arg_13460,
       [size_13449]f32 lifted_2_map2_arg_13461, bool lifted_2_map2_arg_13462,
       f32 lifted_2_map2_arg_13463, bool lifted_2_map2_arg_13464,
       bool lifted_2_map2_arg_13465} =
    lifted_1_map2_8021(size_13417, size_13433, lifted_1_map2_arg_13434,
                       lifted_1_map2_arg_13435, lifted_1_map2_arg_13436,
                       lifted_1_map2_arg_13437, lifted_1_map2_arg_13438,
                       lifted_1_map2_arg_13439, lifted_1_map2_arg_13440,
                       lifted_1_map2_arg_13441, lifted_1_map2_arg_13442,
                       lifted_1_map2_arg_13443, lifted_1_map2_arg_13444,
                       lifted_1_map2_arg_13445, lifted_1_map2_arg_13446,
                       lifted_1_map2_arg_13447, lifted_1_map2_arg_13448,
                       idxs_13420)
  let {[size_13417]f32 vs_13466} =
    lifted_2_map2_8033(size_13417, size_13449, lifted_2_map2_arg_13450,
                       lifted_2_map2_arg_13451, lifted_2_map2_arg_13452,
                       lifted_2_map2_arg_13453, lifted_2_map2_arg_13454,
                       lifted_2_map2_arg_13455, lifted_2_map2_arg_13456,
                       lifted_2_map2_arg_13457, lifted_2_map2_arg_13458,
                       lifted_2_map2_arg_13459, lifted_2_map2_arg_13460,
                       lifted_2_map2_arg_13461, lifted_2_map2_arg_13462,
                       lifted_2_map2_arg_13463, lifted_2_map2_arg_13464,
                       lifted_2_map2_arg_13465, iotas_13432)
  -- vs_13467 aliases vs_13466
  let {[size_13417]f32 vs_13467} = vs_13466
  let {bool lifted_1_segmented_reduce_arg_13468,
       bool lifted_1_segmented_reduce_arg_13469,
       bool lifted_1_segmented_reduce_arg_13470,
       bool lifted_1_segmented_reduce_arg_13471,
       bool lifted_1_segmented_reduce_arg_13472} =
    lifted_0_segmented_reduce_8034(segmented_reduce_13398,
                                   segmented_reduce_13399,
                                   segmented_reduce_13400,
                                   segmented_reduce_13401, op_13397)
  let {bool lifted_2_segmented_reduce_arg_13473,
       bool lifted_2_segmented_reduce_arg_13474,
       f32 lifted_2_segmented_reduce_arg_13475,
       bool lifted_2_segmented_reduce_arg_13476,
       bool lifted_2_segmented_reduce_arg_13477,
       bool lifted_2_segmented_reduce_arg_13478} =
    lifted_1_segmented_reduce_8035(lifted_1_segmented_reduce_arg_13468,
                                   lifted_1_segmented_reduce_arg_13469,
                                   lifted_1_segmented_reduce_arg_13470,
                                   lifted_1_segmented_reduce_arg_13471,
                                   lifted_1_segmented_reduce_arg_13472,
                                   ne_13396)
  -- lifted_3_segmented_reduce_arg_13479 aliases flags_13430
  let {[size_13417]bool lifted_3_segmented_reduce_arg_13479,
       bool lifted_3_segmented_reduce_arg_13480,
       bool lifted_3_segmented_reduce_arg_13481,
       f32 lifted_3_segmented_reduce_arg_13482,
       bool lifted_3_segmented_reduce_arg_13483,
       bool lifted_3_segmented_reduce_arg_13484,
       bool lifted_3_segmented_reduce_arg_13485} =
    lifted_2_segmented_reduce_8036(size_13417,
                                   lifted_2_segmented_reduce_arg_13473,
                                   lifted_2_segmented_reduce_arg_13474,
                                   lifted_2_segmented_reduce_arg_13475,
                                   lifted_2_segmented_reduce_arg_13476,
                                   lifted_2_segmented_reduce_arg_13477,
                                   lifted_2_segmented_reduce_arg_13478,
                                   flags_13430)
  let {i64 size_13486;
       [size_13486]f32 res_13487} =
    lifted_3_segmented_reduce_8074(size_13417,
                                   lifted_3_segmented_reduce_arg_13479,
                                   lifted_3_segmented_reduce_arg_13480,
                                   lifted_3_segmented_reduce_arg_13481,
                                   lifted_3_segmented_reduce_arg_13482,
                                   lifted_3_segmented_reduce_arg_13483,
                                   lifted_3_segmented_reduce_arg_13484,
                                   lifted_3_segmented_reduce_arg_13485,
                                   vs_13467)
  let {i64 ret₂₆_13488} = size_13486
  in {size_13486, res_13487}
}

fun {[n_13489]f32} lifted_4_expand_outer_reduce_8076 (i64 n_13489,
                                                      bool expand_reduce_13490,
                                                      bool expand_reduce_13491,
                                                      bool expand_reduce_13492,
                                                      bool expand_reduce_13493,
                                                      bool expand_reduce_13494,
                                                      bool expand_reduce_13495,
                                                      bool expand_reduce_13496,
                                                      bool get_13497,
                                                      f32 ne_13498,
                                                      bool op_13499,
                                                      bool sz_13500,
                                                      [n_13489]f32 arr_13501,
                                                      [n_13489]f32 arr_13502,
                                                      [n_13489]f32 arr_13503,
                                                      [n_13489]i64 arr_13504,
                                                      [n_13489]f32 arr_13505,
                                                      [n_13489]f32 arr_13506,
                                                      [n_13489]f32 arr_13507,
                                                      [n_13489]f32 arr_13508,
                                                      [n_13489]f32 arr_13509,
                                                      [n_13489]f32 arr_13510,
                                                      [n_13489]f32 arr_13511) = {
  let {bool sz'_13512} = sz_13500
  let {bool get'_13513} = get_13497
  let {f32 get'_13514} = ne_13498
  let {bool get'_13515} = sz_13500
  let {bool lifted_1_expand_reduce_arg_13516,
       bool lifted_1_expand_reduce_arg_13517,
       bool lifted_1_expand_reduce_arg_13518,
       bool lifted_1_expand_reduce_arg_13519,
       bool lifted_1_expand_reduce_arg_13520,
       bool lifted_1_expand_reduce_arg_13521,
       bool lifted_1_expand_reduce_arg_13522,
       bool lifted_1_expand_reduce_arg_13523} =
    lifted_0_expand_reduce_8000(expand_reduce_13490, expand_reduce_13491,
                                expand_reduce_13492, expand_reduce_13493,
                                expand_reduce_13494, expand_reduce_13495,
                                expand_reduce_13496, sz'_13512)
  let {bool lifted_2_expand_reduce_arg_13524,
       f32 lifted_2_expand_reduce_arg_13525,
       bool lifted_2_expand_reduce_arg_13526,
       bool lifted_2_expand_reduce_arg_13527,
       bool lifted_2_expand_reduce_arg_13528,
       bool lifted_2_expand_reduce_arg_13529,
       bool lifted_2_expand_reduce_arg_13530,
       bool lifted_2_expand_reduce_arg_13531,
       bool lifted_2_expand_reduce_arg_13532,
       bool lifted_2_expand_reduce_arg_13533,
       bool lifted_2_expand_reduce_arg_13534} =
    lifted_1_expand_reduce_8001(lifted_1_expand_reduce_arg_13516,
                                lifted_1_expand_reduce_arg_13517,
                                lifted_1_expand_reduce_arg_13518,
                                lifted_1_expand_reduce_arg_13519,
                                lifted_1_expand_reduce_arg_13520,
                                lifted_1_expand_reduce_arg_13521,
                                lifted_1_expand_reduce_arg_13522,
                                lifted_1_expand_reduce_arg_13523, get'_13513,
                                get'_13514, get'_13515)
  let {bool lifted_3_expand_reduce_arg_13535,
       f32 lifted_3_expand_reduce_arg_13536,
       bool lifted_3_expand_reduce_arg_13537,
       bool lifted_3_expand_reduce_arg_13538,
       bool lifted_3_expand_reduce_arg_13539,
       bool lifted_3_expand_reduce_arg_13540,
       bool lifted_3_expand_reduce_arg_13541,
       bool lifted_3_expand_reduce_arg_13542,
       bool lifted_3_expand_reduce_arg_13543,
       bool lifted_3_expand_reduce_arg_13544,
       bool lifted_3_expand_reduce_arg_13545,
       bool lifted_3_expand_reduce_arg_13546} =
    lifted_2_expand_reduce_8002(lifted_2_expand_reduce_arg_13524,
                                lifted_2_expand_reduce_arg_13525,
                                lifted_2_expand_reduce_arg_13526,
                                lifted_2_expand_reduce_arg_13527,
                                lifted_2_expand_reduce_arg_13528,
                                lifted_2_expand_reduce_arg_13529,
                                lifted_2_expand_reduce_arg_13530,
                                lifted_2_expand_reduce_arg_13531,
                                lifted_2_expand_reduce_arg_13532,
                                lifted_2_expand_reduce_arg_13533,
                                lifted_2_expand_reduce_arg_13534, op_13499)
  let {bool lifted_4_expand_reduce_arg_13547,
       f32 lifted_4_expand_reduce_arg_13548,
       bool lifted_4_expand_reduce_arg_13549,
       bool lifted_4_expand_reduce_arg_13550,
       bool lifted_4_expand_reduce_arg_13551,
       bool lifted_4_expand_reduce_arg_13552,
       f32 lifted_4_expand_reduce_arg_13553,
       bool lifted_4_expand_reduce_arg_13554,
       bool lifted_4_expand_reduce_arg_13555,
       bool lifted_4_expand_reduce_arg_13556,
       bool lifted_4_expand_reduce_arg_13557,
       bool lifted_4_expand_reduce_arg_13558,
       bool lifted_4_expand_reduce_arg_13559} =
    lifted_3_expand_reduce_8003(lifted_3_expand_reduce_arg_13535,
                                lifted_3_expand_reduce_arg_13536,
                                lifted_3_expand_reduce_arg_13537,
                                lifted_3_expand_reduce_arg_13538,
                                lifted_3_expand_reduce_arg_13539,
                                lifted_3_expand_reduce_arg_13540,
                                lifted_3_expand_reduce_arg_13541,
                                lifted_3_expand_reduce_arg_13542,
                                lifted_3_expand_reduce_arg_13543,
                                lifted_3_expand_reduce_arg_13544,
                                lifted_3_expand_reduce_arg_13545,
                                lifted_3_expand_reduce_arg_13546, ne_13498)
  -- res_13561 aliases arr_13501, arr_13502, arr_13503, arr_13504, arr_13505, arr_13506, arr_13507, arr_13508, arr_13509, arr_13510, arr_13511
  let {i64 size_13560;
       [size_13560]f32 res_13561} =
    lifted_4_expand_reduce_8075(n_13489, lifted_4_expand_reduce_arg_13547,
                                lifted_4_expand_reduce_arg_13548,
                                lifted_4_expand_reduce_arg_13549,
                                lifted_4_expand_reduce_arg_13550,
                                lifted_4_expand_reduce_arg_13551,
                                lifted_4_expand_reduce_arg_13552,
                                lifted_4_expand_reduce_arg_13553,
                                lifted_4_expand_reduce_arg_13554,
                                lifted_4_expand_reduce_arg_13555,
                                lifted_4_expand_reduce_arg_13556,
                                lifted_4_expand_reduce_arg_13557,
                                lifted_4_expand_reduce_arg_13558,
                                lifted_4_expand_reduce_arg_13559, arr_13501,
                                arr_13502, arr_13503, arr_13504, arr_13505,
                                arr_13506, arr_13507, arr_13508, arr_13509,
                                arr_13510, arr_13511)
  let {i64 ret₁₁_13562} = size_13560
  let {bool dim_match_13563} = eq_i64(n_13489, size_13560)
  let {cert empty_or_match_cert_13564} =
    assert(dim_match_13563, "Value of (core language) shape (", size_13560,
                            ") cannot match shape of type `[", n_13489, "]b`.",
           "lib/github.com/diku-dk/segmented/segmented.fut:103:6-45")
  -- res_13565 aliases res_13561
  let {[n_13489]f32 res_13565} =
    <empty_or_match_cert_13564>
    reshape((~n_13489), res_13561)
  in {res_13565}
}

fun {bool, bool} lifted_2_map_8077 (i64 d_13566, i64 d_13567, bool f_13568,
                                    bool f_13569) = {
  {f_13568, f_13569}
}

fun {bool} lifted_1_map_8079 (i64 d_13570, bool f_13571) = {
  {f_13571}
}

fun {bool} lifted_0_reduce_8083 (bool nameless_13572, bool op_13573) = {
  {op_13573}
}

fun {f32, bool} lifted_1_reduce_8084 (bool op_13574, f32 ne_13575) = {
  {ne_13575, op_13574}
}

fun {f32} lifted_0_op_8087 (bool nameless_13576, f32 x_13577) = {
  {x_13577}
}

fun {f32} lifted_1_op_8088 (f32 x_13578, f32 x_13579) = {
  let {f32 res_13580} = fadd32(x_13578, x_13579)
  in {res_13580}
}

fun {f32} lifted_2_reduce_8089 (i64 n_13581, f32 ne_13582, bool op_13583,
                                [n_13581]f32 as_13584) = {
  let {f32 res_13585} =
    redomap(n_13581,
            {fn {f32} (f32 x_13586, f32 x_13587) =>
               let {f32 lifted_1_op_arg_13588} =
                 lifted_0_op_8087(op_13583, x_13586)
               let {f32 res_13589} =
                 lifted_1_op_8088(lifted_1_op_arg_13588, x_13587)
               in {res_13589},
             {ne_13582}},
            fn {f32} (f32 x_13590) =>
              {x_13590},
            as_13584)
  in {res_13585}
}

fun {f32} lifted_0_f_8090 (i64 n_13591, bool reduce_13592,
                           [n_13591]f32 x_13593) = {
  let {bool lifted_1_reduce_arg_13594} =
    lifted_0_reduce_8083(reduce_13592, true)
  let {f32 lifted_2_reduce_arg_13595, bool lifted_2_reduce_arg_13596} =
    lifted_1_reduce_8084(lifted_1_reduce_arg_13594, 0.0f32)
  let {f32 res_13597} =
    lifted_2_reduce_8089(n_13591, lifted_2_reduce_arg_13595,
                         lifted_2_reduce_arg_13596, x_13593)
  in {res_13597}
}

fun {*[n_13598]f32} lifted_2_map_8091 (i64 n_13598, i64 d_13599, bool f_13600,
                                       [n_13598][d_13599]f32 as_13601) = {
  let {[n_13598]f32 res_13602} =
    map(n_13598,
        fn {f32} ([d_13599]f32 x_13603) =>
          let {f32 res_13604} =
            lifted_0_f_8090(d_13599, f_13600, x_13603)
          in {res_13604},
        as_13601)
  let {i64 ret₂_13605} = n_13598
  in {res_13602}
}

fun {bool} lifted_0_map_8092 (bool nameless_13606, bool f_13607) = {
  {f_13607}
}

fun {f32} lifted_0_f_8094 (bool nameless_13608, f32 x_13609) = {
  let {f32 res_13610} =
    max_3281(0.0f32, x_13609)
  in {res_13610}
}

fun {*[n_13611]f32} lifted_1_map_8095 (i64 n_13611, bool f_13612,
                                       [n_13611]f32 as_13613) = {
  let {[n_13611]f32 res_13614} =
    map(n_13611,
        fn {f32} (f32 x_13615) =>
          let {f32 res_13616} =
            lifted_0_f_8094(f_13612, x_13615)
          in {res_13616},
        as_13613)
  let {i64 ret₂_13617} = n_13611
  in {res_13614}
}

fun {bool} lifted_0_reduce_8098 (bool nameless_13618, bool op_13619) = {
  {op_13619}
}

fun {f32, bool} lifted_1_reduce_8099 (bool op_13620, f32 ne_13621) = {
  {ne_13621, op_13620}
}

fun {f32} lifted_0_op_8102 (bool nameless_13622, f32 x_13623) = {
  {x_13623}
}

fun {f32} lifted_1_op_8103 (f32 x_13624, f32 x_13625) = {
  let {f32 res_13626} = fadd32(x_13624, x_13625)
  in {res_13626}
}

fun {f32} lifted_2_reduce_8104 (i64 n_13627, f32 ne_13628, bool op_13629,
                                [n_13627]f32 as_13630) = {
  let {f32 res_13631} =
    redomap(n_13627,
            {fn {f32} (f32 x_13632, f32 x_13633) =>
               let {f32 lifted_1_op_arg_13634} =
                 lifted_0_op_8102(op_13629, x_13632)
               let {f32 res_13635} =
                 lifted_1_op_8103(lifted_1_op_arg_13634, x_13633)
               in {res_13635},
             {ne_13628}},
            fn {f32} (f32 x_13636) =>
              {x_13636},
            as_13630)
  in {res_13631}
}

fun {f32} lifted_0_f_8105 (i64 n_13637, i64 paths_13638, bool map_13639,
                           bool reduce_13640,
                           [paths_13638][n_13637]f32 xs_13641) = {
  let {i64 lifted_1_map_arg_13642} =
    map_7820(n_13637)
  let {bool lifted_2_map_arg_13643} =
    lifted_1_map_8079(lifted_1_map_arg_13642, reduce_13640)
  let {[paths_13638]f32 netted_13644} =
    lifted_2_map_8091(paths_13638, n_13637, lifted_2_map_arg_13643, xs_13641)
  -- netted_13645 aliases netted_13644
  let {[paths_13638]f32 netted_13645} = netted_13644
  let {bool lifted_1_map_arg_13646} =
    lifted_0_map_8092(map_13639, true)
  let {[paths_13638]f32 pfe_13647} =
    lifted_1_map_8095(paths_13638, lifted_1_map_arg_13646, netted_13645)
  -- pfe_13648 aliases pfe_13647
  let {[paths_13638]f32 pfe_13648} = pfe_13647
  let {bool lifted_1_reduce_arg_13649} =
    lifted_0_reduce_8098(reduce_13640, true)
  let {f32 lifted_2_reduce_arg_13650, bool lifted_2_reduce_arg_13651} =
    lifted_1_reduce_8099(lifted_1_reduce_arg_13649, 0.0f32)
  let {f32 x_13652} =
    lifted_2_reduce_8104(paths_13638, lifted_2_reduce_arg_13650,
                         lifted_2_reduce_arg_13651, pfe_13648)
  let {f32 y_13653} =
    i64_3244(paths_13638)
  let {f32 res_13654} = fdiv32(x_13652, y_13653)
  in {res_13654}
}

fun {*[n_13655]f32} lifted_3_map_8106 (i64 n_13655, i64 d_13656, i64 d_13657,
                                       bool f_13658, bool f_13659,
                                       [n_13655][d_13656][d_13657]f32 as_13660) = {
  let {[n_13655]f32 res_13661} =
    map(n_13655,
        fn {f32} ([d_13656][d_13657]f32 x_13662) =>
          let {f32 res_13663} =
            lifted_0_f_8105(d_13657, d_13656, f_13658, f_13659, x_13662)
          in {res_13663},
        as_13660)
  let {i64 ret₂_13664} = n_13655
  in {res_13661}
}

fun {f32, f32, f32, f32, f32, bool} lifted_0_map2_8107 (bool map_13665,
                                                        f32 f_13666,
                                                        f32 f_13667,
                                                        f32 f_13668,
                                                        f32 f_13669,
                                                        f32 f_13670) = {
  {f_13666, f_13667, f_13668, f_13669, f_13670, map_13665}
}

fun {[n_13671]f32, f32, f32, f32, f32, f32, bool}
lifted_1_map2_8108 (i64 n_13671, f32 f_13672, f32 f_13673, f32 f_13674,
                    f32 f_13675, f32 f_13676, bool map_13677,
                    [n_13671]f32 as_13678) = {
  {as_13678, f_13672, f_13673, f_13674, f_13675, f_13676, map_13677}
}

fun {f32, f32, f32, f32, f32} lifted_0_map_8109 (bool nameless_13679,
                                                 f32 f_13680, f32 f_13681,
                                                 f32 f_13682, f32 f_13683,
                                                 f32 f_13684) = {
  {f_13680, f_13681, f_13682, f_13683, f_13684}
}

fun {f32, f32, f32, f32, f32, f32} lifted_0_f_8111 (f32 a_13685, f32 b_13686,
                                                    f32 deltat_13687,
                                                    f32 r0_13688,
                                                    f32 sigma_13689,
                                                    f32 y_13690) = {
  {a_13685, b_13686, deltat_13687, r0_13688, sigma_13689, y_13690}
}

fun {f32} lifted_1_f_8112 (f32 a_13691, f32 b_13692, f32 deltat_13693,
                           f32 r0_13694, f32 sigma_13695, f32 y_13696,
                           f32 z_13697) = {
  let {f32 y_13698} =
    bondprice_7156(a_13691, b_13692, deltat_13693, r0_13694, sigma_13695,
                   5.0e-2f32, 0.0f32, z_13697)
  let {f32 res_13699} = fmul32(y_13696, y_13698)
  in {res_13699}
}

fun {f32} lifted_0_f_8113 (f32 f_13700, f32 f_13701, f32 f_13702, f32 f_13703,
                           f32 f_13704, f32 a_13705, f32 b_13706) = {
  let {f32 lifted_1_f_arg_13707, f32 lifted_1_f_arg_13708,
       f32 lifted_1_f_arg_13709, f32 lifted_1_f_arg_13710,
       f32 lifted_1_f_arg_13711, f32 lifted_1_f_arg_13712} =
    lifted_0_f_8111(f_13700, f_13701, f_13702, f_13703, f_13704, a_13705)
  let {f32 res_13713} =
    lifted_1_f_8112(lifted_1_f_arg_13707, lifted_1_f_arg_13708,
                    lifted_1_f_arg_13709, lifted_1_f_arg_13710,
                    lifted_1_f_arg_13711, lifted_1_f_arg_13712, b_13706)
  in {res_13713}
}

fun {*[n_13714]f32} lifted_1_map_8114 (i64 n_13714, f32 f_13715, f32 f_13716,
                                       f32 f_13717, f32 f_13718, f32 f_13719,
                                       [n_13714]f32 as_13720,
                                       [n_13714]f32 as_13721) = {
  let {[n_13714]f32 res_13722} =
    map(n_13714,
        fn {f32} (f32 x_13723, f32 x_13724) =>
          let {f32 res_13725} =
            lifted_0_f_8113(f_13715, f_13716, f_13717, f_13718, f_13719,
                            x_13723, x_13724)
          in {res_13725},
        as_13720, as_13721)
  let {i64 ret₂_13726} = n_13714
  in {res_13722}
}

fun {*[n_13727]f32} lifted_2_map2_8115 (i64 n_13727, [n_13727]f32 as_13728,
                                        f32 f_13729, f32 f_13730, f32 f_13731,
                                        f32 f_13732, f32 f_13733,
                                        bool map_13734,
                                        [n_13727]f32 bs_13735) = {
  -- lifted_1_map_arg_13736 aliases as_13728, bs_13735
  -- lifted_1_map_arg_13737 aliases as_13728, bs_13735
  let {[n_13727]f32 lifted_1_map_arg_13736,
       [n_13727]f32 lifted_1_map_arg_13737} =
    zip2_7734(n_13727, as_13728, bs_13735)
  let {f32 lifted_1_map_arg_13738, f32 lifted_1_map_arg_13739,
       f32 lifted_1_map_arg_13740, f32 lifted_1_map_arg_13741,
       f32 lifted_1_map_arg_13742} =
    lifted_0_map_8109(map_13734, f_13729, f_13730, f_13731, f_13732, f_13733)
  let {[n_13727]f32 res_13743} =
    lifted_1_map_8114(n_13727, lifted_1_map_arg_13738, lifted_1_map_arg_13739,
                      lifted_1_map_arg_13740, lifted_1_map_arg_13741,
                      lifted_1_map_arg_13742, lifted_1_map_arg_13736,
                      lifted_1_map_arg_13737)
  in {res_13743}
}

fun {bool} lifted_0_reduce_8118 (bool nameless_13744, bool op_13745) = {
  {op_13745}
}

fun {f32, bool} lifted_1_reduce_8119 (bool op_13746, f32 ne_13747) = {
  {ne_13747, op_13746}
}

fun {f32} lifted_0_op_8122 (bool nameless_13748, f32 x_13749) = {
  {x_13749}
}

fun {f32} lifted_1_op_8123 (f32 x_13750, f32 x_13751) = {
  let {f32 res_13752} = fadd32(x_13750, x_13751)
  in {res_13752}
}

fun {f32} lifted_2_reduce_8124 (i64 n_13753, f32 ne_13754, bool op_13755,
                                [n_13753]f32 as_13756) = {
  let {f32 res_13757} =
    redomap(n_13753,
            {fn {f32} (f32 x_13758, f32 x_13759) =>
               let {f32 lifted_1_op_arg_13760} =
                 lifted_0_op_8122(op_13755, x_13758)
               let {f32 res_13761} =
                 lifted_1_op_8123(lifted_1_op_arg_13760, x_13759)
               in {res_13761},
             {ne_13754}},
            fn {f32} (f32 x_13762) =>
              {x_13762},
            as_13756)
  in {res_13757}
}

fun {f32, *[steps_13765]f32} main_7823 (i64 n_13763, i64 paths_13764,
                                        i64 steps_13765,
                                        [n_13763]f32 swap_term_13766,
                                        [n_13763]i64 payments_13767,
                                        [n_13763]f32 notional_13768,
                                        f32 a_13769, f32 b_13770,
                                        f32 sigma_13771, f32 r0_13772) = {
  let {bool lifted_1_map2_arg_13773, bool lifted_1_map2_arg_13774} =
    lifted_0_map2_7952(true, true)
  -- lifted_2_map2_arg_13775 aliases swap_term_13766
  let {[n_13763]f32 lifted_2_map2_arg_13775, bool lifted_2_map2_arg_13776,
       bool lifted_2_map2_arg_13777} =
    lifted_1_map2_7953(n_13763, lifted_1_map2_arg_13773,
                       lifted_1_map2_arg_13774, swap_term_13766)
  let {[n_13763]f32 maximum_arg_13778} =
    lifted_2_map2_7960(n_13763, lifted_2_map2_arg_13775,
                       lifted_2_map2_arg_13776, lifted_2_map2_arg_13777,
                       payments_13767)
  let {f32 max_duration_13779} =
    maximum_7678(n_13763, maximum_arg_13778)
  let {f32 max_duration_13780} = max_duration_13779
  let {f32 y_13781} =
    i64_3244(steps_13765)
  let {f32 dt_13782} = fdiv32(max_duration_13780, y_13781)
  let {f32 dt_13783} = dt_13782
  let {f32 a_13784} = a_13769
  let {f32 b_13785} = b_13770
  let {f32 deltat_13786} = dt_13783
  let {f32 r0_13787} = r0_13772
  let {f32 sigma_13788} = sigma_13771
  let {[n_13763]i64 lifted_1_map_arg_13789} =
    indices_7702(n_13763, swap_term_13766)
  -- lifted_1_map_arg_13796 aliases swap_term_13766, payments_13767, notional_13768
  -- lifted_1_map_arg_13797 aliases swap_term_13766, payments_13767, notional_13768
  -- lifted_1_map_arg_13800 aliases swap_term_13766, payments_13767, notional_13768
  let {i64 size_13790, i64 size_13791, i64 size_13792;
       f32 lifted_1_map_arg_13793, f32 lifted_1_map_arg_13794,
       f32 lifted_1_map_arg_13795, [size_13790]f32 lifted_1_map_arg_13796,
       [size_13791]i64 lifted_1_map_arg_13797, f32 lifted_1_map_arg_13798,
       f32 lifted_1_map_arg_13799, [size_13792]f32 lifted_1_map_arg_13800} =
    lifted_0_map_7961(n_13763, true, a_13784, b_13785, deltat_13786,
                      notional_13768, payments_13767, r0_13787, sigma_13788,
                      swap_term_13766)
  let {bool dim_match_13801} = eq_i64(size_13790, size_13791)
  let {cert empty_or_match_cert_13802} =
    assert(dim_match_13801, "function arguments of wrong shape",
           "cva.fut:112:17-116:85")
  -- lifted_1_map_arg_13803 aliases lifted_1_map_arg_13797
  let {[size_13790]i64 lifted_1_map_arg_13803} =
    <empty_or_match_cert_13802>
    reshape((~size_13790), lifted_1_map_arg_13797)
  let {bool dim_match_13804} = eq_i64(size_13790, size_13792)
  let {cert empty_or_match_cert_13805} =
    assert(dim_match_13804, "function arguments of wrong shape",
           "cva.fut:112:17-116:85")
  -- lifted_1_map_arg_13806 aliases lifted_1_map_arg_13800
  let {[size_13790]f32 lifted_1_map_arg_13806} =
    <empty_or_match_cert_13805>
    reshape((~size_13790), lifted_1_map_arg_13800)
  let {[n_13763]f32 swaps_13807, [n_13763]f32 swaps_13808,
       [n_13763]i64 swaps_13809, [n_13763]f32 swaps_13810} =
    lifted_1_map_7964(n_13763, size_13790, lifted_1_map_arg_13793,
                      lifted_1_map_arg_13794, lifted_1_map_arg_13795,
                      lifted_1_map_arg_13796, lifted_1_map_arg_13803,
                      lifted_1_map_arg_13798, lifted_1_map_arg_13799,
                      lifted_1_map_arg_13806, lifted_1_map_arg_13789)
  -- swaps_13811 aliases swaps_13807
  let {[n_13763]f32 swaps_13811} = swaps_13807
  -- swaps_13812 aliases swaps_13808
  let {[n_13763]f32 swaps_13812} = swaps_13808
  -- swaps_13813 aliases swaps_13809
  let {[n_13763]i64 swaps_13813} = swaps_13809
  -- swaps_13814 aliases swaps_13810
  let {[n_13763]f32 swaps_13814} = swaps_13810
  let {[steps_13765]f32 times_13815} =
    gen_times_7248(steps_13765, max_duration_13780)
  -- times_13816 aliases times_13815
  let {[steps_13765]f32 times_13816} = times_13815
  let {i64 to_i64_13817} = sext i32 0i32 to i64
  let {bool x_13818} = sle64(0i64, to_i64_13817)
  let {bool y_13819} = slt64(to_i64_13817, n_13763)
  let {bool bounds_check_13820} = logand(x_13818, y_13819)
  let {cert index_certs_13821} =
    assert(bounds_check_13820, "Index [", to_i64_13817,
                               "] out of bounds for array of shape [", n_13763,
                               "].", "cva.fut:119:21-32")
  let {f32 x_13822} =
    <index_certs_13821>
    swap_term_13766[to_i64_13817]
  let {i64 to_i64_13823} = sext i32 0i32 to i64
  let {bool x_13824} = sle64(0i64, to_i64_13823)
  let {bool y_13825} = slt64(to_i64_13823, n_13763)
  let {bool bounds_check_13826} = logand(x_13824, y_13825)
  let {cert index_certs_13827} =
    assert(bounds_check_13826, "Index [", to_i64_13823,
                               "] out of bounds for array of shape [", n_13763,
                               "].", "cva.fut:119:44-54")
  let {i64 x_13828} =
    <index_certs_13827>
    payments_13767[to_i64_13823]
  let {i64 i64_arg_13829} = sub64(x_13828, 1i64)
  let {f32 y_13830} =
    i64_3244(i64_arg_13829)
  let {f32 last_date_13831} = fmul32(x_13822, y_13830)
  let {f32 last_date_13832} = last_date_13831
  let {[1i64]i32 rng_from_seed_arg_13833} = [1i32]
  let {i32 rng_13834} =
    rng_from_seed_7704(1i64, rng_from_seed_arg_13833)
  let {i32 rng_13835} = rng_13834
  let {[paths_13764]i32 rng_vec_13836} =
    split_rng_7575(paths_13764, rng_13835)
  -- rng_vec_13837 aliases rng_vec_13836
  let {[paths_13764]i32 rng_vec_13837} = rng_vec_13836
  let {i64 lifted_1_map_arg_13838} =
    map_7709(steps_13765)
  let {i64 lifted_2_map_arg_13839, bool lifted_2_map_arg_13840,
       i64 lifted_2_map_arg_13841} =
    lifted_1_map_7965(lifted_1_map_arg_13838, true, steps_13765)
  let {[paths_13764][lifted_2_map_arg_13839]f32 rands_13842} =
    lifted_2_map_7972(paths_13764, lifted_2_map_arg_13839,
                      lifted_2_map_arg_13840, lifted_2_map_arg_13841,
                      rng_vec_13837)
  -- rands_13843 aliases rands_13842
  let {[paths_13764][lifted_2_map_arg_13839]f32 rands_13843} = rands_13842
  let {i64 lifted_2_map_arg_13844, i64 lifted_2_map_arg_13845} =
    map_7713(steps_13765, steps_13765)
  let {i64 lifted_3_map_arg_13846, f32 lifted_3_map_arg_13847,
       f32 lifted_3_map_arg_13848, f32 lifted_3_map_arg_13849,
       f32 lifted_3_map_arg_13850, f32 lifted_3_map_arg_13851,
       f32 lifted_3_map_arg_13852} =
    lifted_2_map_7973(lifted_2_map_arg_13844, lifted_2_map_arg_13845, a_13784,
                      b_13785, deltat_13786, r0_13772, r0_13787, sigma_13788)
  let {[paths_13764][lifted_3_map_arg_13846]f32 shortrates_13853} =
    lifted_3_map_7976(paths_13764, lifted_2_map_arg_13839,
                      lifted_3_map_arg_13846, lifted_3_map_arg_13847,
                      lifted_3_map_arg_13848, lifted_3_map_arg_13849,
                      lifted_3_map_arg_13850, lifted_3_map_arg_13851,
                      lifted_3_map_arg_13852, rands_13843)
  -- shortrates_13854 aliases shortrates_13853
  let {[paths_13764][lifted_3_map_arg_13846]f32 shortrates_13854} =
    shortrates_13853
  let {i64 lifted_3_map_arg_13855, i64 lifted_3_map_arg_13856,
       i64 lifted_3_map_arg_13857} =
    map_7729(steps_13765, steps_13765, n_13763)
  -- lifted_4_map_arg_13868 aliases swaps_13811, swaps_13812, swaps_13813, swaps_13814, times_13816
  -- lifted_4_map_arg_13869 aliases swaps_13811, swaps_13812, swaps_13813, swaps_13814, times_13816
  -- lifted_4_map_arg_13870 aliases swaps_13811, swaps_13812, swaps_13813, swaps_13814, times_13816
  -- lifted_4_map_arg_13871 aliases swaps_13811, swaps_13812, swaps_13813, swaps_13814, times_13816
  -- lifted_4_map_arg_13872 aliases swaps_13811, swaps_13812, swaps_13813, swaps_13814, times_13816
  let {i64 size_13858, i64 size_13859;
       i64 lifted_4_map_arg_13860, i64 lifted_4_map_arg_13861,
       f32 lifted_4_map_arg_13862, f32 lifted_4_map_arg_13863,
       f32 lifted_4_map_arg_13864, bool lifted_4_map_arg_13865,
       f32 lifted_4_map_arg_13866, f32 lifted_4_map_arg_13867,
       [size_13858]f32 lifted_4_map_arg_13868,
       [size_13858]f32 lifted_4_map_arg_13869,
       [size_13858]i64 lifted_4_map_arg_13870,
       [size_13858]f32 lifted_4_map_arg_13871,
       [size_13859]f32 lifted_4_map_arg_13872} =
    lifted_3_map_7977(n_13763, steps_13765, lifted_3_map_arg_13855,
                      lifted_3_map_arg_13856, lifted_3_map_arg_13857, a_13784,
                      b_13785, deltat_13786, true, r0_13787, sigma_13788,
                      swaps_13811, swaps_13812, swaps_13813, swaps_13814,
                      times_13816)
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13873,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13874,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13875,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]i64 pricings_13876,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13877,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13878,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13879,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13880,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13881,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13882,
       [paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13883} =
    lifted_4_map_7993(paths_13764, lifted_3_map_arg_13846, size_13858,
                      size_13859, lifted_4_map_arg_13860,
                      lifted_4_map_arg_13861, lifted_4_map_arg_13862,
                      lifted_4_map_arg_13863, lifted_4_map_arg_13864,
                      lifted_4_map_arg_13865, lifted_4_map_arg_13866,
                      lifted_4_map_arg_13867, lifted_4_map_arg_13868,
                      lifted_4_map_arg_13869, lifted_4_map_arg_13870,
                      lifted_4_map_arg_13871, lifted_4_map_arg_13872,
                      shortrates_13854)
  -- pricings_13884 aliases pricings_13873
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13884} =
    pricings_13873
  -- pricings_13885 aliases pricings_13874
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13885} =
    pricings_13874
  -- pricings_13886 aliases pricings_13875
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13886} =
    pricings_13875
  -- pricings_13887 aliases pricings_13876
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]i64 pricings_13887} =
    pricings_13876
  -- pricings_13888 aliases pricings_13877
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13888} =
    pricings_13877
  -- pricings_13889 aliases pricings_13878
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13889} =
    pricings_13878
  -- pricings_13890 aliases pricings_13879
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13890} =
    pricings_13879
  -- pricings_13891 aliases pricings_13880
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13891} =
    pricings_13880
  -- pricings_13892 aliases pricings_13881
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13892} =
    pricings_13881
  -- pricings_13893 aliases pricings_13882
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13893} =
    pricings_13882
  -- pricings_13894 aliases pricings_13883
  let {[paths_13764][lifted_4_map_arg_13860][lifted_4_map_arg_13861]f32 pricings_13894} =
    pricings_13883
  -- flattened_13896 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13897 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13898 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13899 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13900 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13901 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13902 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13903 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13904 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13905 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  -- flattened_13906 aliases pricings_13884, pricings_13885, pricings_13886, pricings_13887, pricings_13888, pricings_13889, pricings_13890, pricings_13891, pricings_13892, pricings_13893, pricings_13894
  let {i64 size_13895;
       [size_13895]f32 flattened_13896, [size_13895]f32 flattened_13897,
       [size_13895]f32 flattened_13898, [size_13895]i64 flattened_13899,
       [size_13895]f32 flattened_13900, [size_13895]f32 flattened_13901,
       [size_13895]f32 flattened_13902, [size_13895]f32 flattened_13903,
       [size_13895]f32 flattened_13904, [size_13895]f32 flattened_13905,
       [size_13895]f32 flattened_13906} =
    flatten_3d_7744(paths_13764, lifted_4_map_arg_13860, lifted_4_map_arg_13861,
                    pricings_13884, pricings_13885, pricings_13886,
                    pricings_13887, pricings_13888, pricings_13889,
                    pricings_13890, pricings_13891, pricings_13892,
                    pricings_13893, pricings_13894)
  let {i64 ret₆₂_13907} = size_13895
  -- flattened_13908 aliases flattened_13896
  let {[size_13895]f32 flattened_13908} = flattened_13896
  -- flattened_13909 aliases flattened_13897
  let {[size_13895]f32 flattened_13909} = flattened_13897
  -- flattened_13910 aliases flattened_13898
  let {[size_13895]f32 flattened_13910} = flattened_13898
  -- flattened_13911 aliases flattened_13899
  let {[size_13895]i64 flattened_13911} = flattened_13899
  -- flattened_13912 aliases flattened_13900
  let {[size_13895]f32 flattened_13912} = flattened_13900
  -- flattened_13913 aliases flattened_13901
  let {[size_13895]f32 flattened_13913} = flattened_13901
  -- flattened_13914 aliases flattened_13902
  let {[size_13895]f32 flattened_13914} = flattened_13902
  -- flattened_13915 aliases flattened_13903
  let {[size_13895]f32 flattened_13915} = flattened_13903
  -- flattened_13916 aliases flattened_13904
  let {[size_13895]f32 flattened_13916} = flattened_13904
  -- flattened_13917 aliases flattened_13905
  let {[size_13895]f32 flattened_13917} = flattened_13905
  -- flattened_13918 aliases flattened_13906
  let {[size_13895]f32 flattened_13918} = flattened_13906
  let {bool lifted_1_expand_outer_reduce_arg_13919,
       bool lifted_1_expand_outer_reduce_arg_13920,
       bool lifted_1_expand_outer_reduce_arg_13921,
       bool lifted_1_expand_outer_reduce_arg_13922,
       bool lifted_1_expand_outer_reduce_arg_13923,
       bool lifted_1_expand_outer_reduce_arg_13924,
       bool lifted_1_expand_outer_reduce_arg_13925,
       bool lifted_1_expand_outer_reduce_arg_13926} =
    lifted_0_expand_outer_reduce_7994(true, true, true, true, true, true, true,
                                      true)
  let {bool lifted_2_expand_outer_reduce_arg_13927,
       bool lifted_2_expand_outer_reduce_arg_13928,
       bool lifted_2_expand_outer_reduce_arg_13929,
       bool lifted_2_expand_outer_reduce_arg_13930,
       bool lifted_2_expand_outer_reduce_arg_13931,
       bool lifted_2_expand_outer_reduce_arg_13932,
       bool lifted_2_expand_outer_reduce_arg_13933,
       bool lifted_2_expand_outer_reduce_arg_13934,
       bool lifted_2_expand_outer_reduce_arg_13935} =
    lifted_1_expand_outer_reduce_7995(lifted_1_expand_outer_reduce_arg_13919,
                                      lifted_1_expand_outer_reduce_arg_13920,
                                      lifted_1_expand_outer_reduce_arg_13921,
                                      lifted_1_expand_outer_reduce_arg_13922,
                                      lifted_1_expand_outer_reduce_arg_13923,
                                      lifted_1_expand_outer_reduce_arg_13924,
                                      lifted_1_expand_outer_reduce_arg_13925,
                                      lifted_1_expand_outer_reduce_arg_13926,
                                      true)
  let {bool lifted_3_expand_outer_reduce_arg_13936,
       bool lifted_3_expand_outer_reduce_arg_13937,
       bool lifted_3_expand_outer_reduce_arg_13938,
       bool lifted_3_expand_outer_reduce_arg_13939,
       bool lifted_3_expand_outer_reduce_arg_13940,
       bool lifted_3_expand_outer_reduce_arg_13941,
       bool lifted_3_expand_outer_reduce_arg_13942,
       bool lifted_3_expand_outer_reduce_arg_13943,
       bool lifted_3_expand_outer_reduce_arg_13944,
       bool lifted_3_expand_outer_reduce_arg_13945} =
    lifted_2_expand_outer_reduce_7998(lifted_2_expand_outer_reduce_arg_13927,
                                      lifted_2_expand_outer_reduce_arg_13928,
                                      lifted_2_expand_outer_reduce_arg_13929,
                                      lifted_2_expand_outer_reduce_arg_13930,
                                      lifted_2_expand_outer_reduce_arg_13931,
                                      lifted_2_expand_outer_reduce_arg_13932,
                                      lifted_2_expand_outer_reduce_arg_13933,
                                      lifted_2_expand_outer_reduce_arg_13934,
                                      lifted_2_expand_outer_reduce_arg_13935,
                                      true)
  let {bool lifted_4_expand_outer_reduce_arg_13946,
       bool lifted_4_expand_outer_reduce_arg_13947,
       bool lifted_4_expand_outer_reduce_arg_13948,
       bool lifted_4_expand_outer_reduce_arg_13949,
       bool lifted_4_expand_outer_reduce_arg_13950,
       bool lifted_4_expand_outer_reduce_arg_13951,
       bool lifted_4_expand_outer_reduce_arg_13952,
       bool lifted_4_expand_outer_reduce_arg_13953,
       f32 lifted_4_expand_outer_reduce_arg_13954,
       bool lifted_4_expand_outer_reduce_arg_13955,
       bool lifted_4_expand_outer_reduce_arg_13956} =
    lifted_3_expand_outer_reduce_7999(lifted_3_expand_outer_reduce_arg_13936,
                                      lifted_3_expand_outer_reduce_arg_13937,
                                      lifted_3_expand_outer_reduce_arg_13938,
                                      lifted_3_expand_outer_reduce_arg_13939,
                                      lifted_3_expand_outer_reduce_arg_13940,
                                      lifted_3_expand_outer_reduce_arg_13941,
                                      lifted_3_expand_outer_reduce_arg_13942,
                                      lifted_3_expand_outer_reduce_arg_13943,
                                      lifted_3_expand_outer_reduce_arg_13944,
                                      lifted_3_expand_outer_reduce_arg_13945,
                                      0.0f32)
  -- prices_13957 aliases flattened_13908, flattened_13909, flattened_13910, flattened_13911, flattened_13912, flattened_13913, flattened_13914, flattened_13915, flattened_13916, flattened_13917, flattened_13918
  let {[size_13895]f32 prices_13957} =
    lifted_4_expand_outer_reduce_8076(size_13895,
                                      lifted_4_expand_outer_reduce_arg_13946,
                                      lifted_4_expand_outer_reduce_arg_13947,
                                      lifted_4_expand_outer_reduce_arg_13948,
                                      lifted_4_expand_outer_reduce_arg_13949,
                                      lifted_4_expand_outer_reduce_arg_13950,
                                      lifted_4_expand_outer_reduce_arg_13951,
                                      lifted_4_expand_outer_reduce_arg_13952,
                                      lifted_4_expand_outer_reduce_arg_13953,
                                      lifted_4_expand_outer_reduce_arg_13954,
                                      lifted_4_expand_outer_reduce_arg_13955,
                                      lifted_4_expand_outer_reduce_arg_13956,
                                      flattened_13908, flattened_13909,
                                      flattened_13910, flattened_13911,
                                      flattened_13912, flattened_13913,
                                      flattened_13914, flattened_13915,
                                      flattened_13916, flattened_13917,
                                      flattened_13918)
  -- prices_13958 aliases prices_13957
  let {[size_13895]f32 prices_13958} = prices_13957
  -- unflattened_13959 aliases prices_13958
  let {[paths_13764][steps_13765][n_13763]f32 unflattened_13959} =
    unflatten_3d_7813(size_13895, paths_13764, steps_13765, n_13763,
                      prices_13958)
  -- unflattened_13960 aliases unflattened_13959
  let {[paths_13764][steps_13765][n_13763]f32 unflattened_13960} =
    unflattened_13959
  -- transposed_13961 aliases unflattened_13960
  let {[steps_13765][paths_13764][n_13763]f32 transposed_13961} =
    transpose_7815(paths_13764, steps_13765, n_13763, unflattened_13960)
  -- transposed_13962 aliases transposed_13961
  let {[steps_13765][paths_13764][n_13763]f32 transposed_13962} =
    transposed_13961
  let {i64 lifted_2_map_arg_13963, i64 lifted_2_map_arg_13964} =
    map_7818(paths_13764, n_13763)
  let {bool lifted_3_map_arg_13965, bool lifted_3_map_arg_13966} =
    lifted_2_map_8077(lifted_2_map_arg_13963, lifted_2_map_arg_13964, true,
                      true)
  let {[steps_13765]f32 avgexp_13967} =
    lifted_3_map_8106(steps_13765, paths_13764, n_13763, lifted_3_map_arg_13965,
                      lifted_3_map_arg_13966, transposed_13962)
  -- avgexp_13968 aliases avgexp_13967
  let {[steps_13765]f32 avgexp_13968} = avgexp_13967
  let {f32 lifted_1_map2_arg_13969, f32 lifted_1_map2_arg_13970,
       f32 lifted_1_map2_arg_13971, f32 lifted_1_map2_arg_13972,
       f32 lifted_1_map2_arg_13973, bool lifted_1_map2_arg_13974} =
    lifted_0_map2_8107(true, a_13784, b_13785, deltat_13786, r0_13787,
                       sigma_13788)
  -- lifted_2_map2_arg_13975 aliases avgexp_13968
  let {[steps_13765]f32 lifted_2_map2_arg_13975, f32 lifted_2_map2_arg_13976,
       f32 lifted_2_map2_arg_13977, f32 lifted_2_map2_arg_13978,
       f32 lifted_2_map2_arg_13979, f32 lifted_2_map2_arg_13980,
       bool lifted_2_map2_arg_13981} =
    lifted_1_map2_8108(steps_13765, lifted_1_map2_arg_13969,
                       lifted_1_map2_arg_13970, lifted_1_map2_arg_13971,
                       lifted_1_map2_arg_13972, lifted_1_map2_arg_13973,
                       lifted_1_map2_arg_13974, avgexp_13968)
  let {[steps_13765]f32 dexp_13982} =
    lifted_2_map2_8115(steps_13765, lifted_2_map2_arg_13975,
                       lifted_2_map2_arg_13976, lifted_2_map2_arg_13977,
                       lifted_2_map2_arg_13978, lifted_2_map2_arg_13979,
                       lifted_2_map2_arg_13980, lifted_2_map2_arg_13981,
                       times_13816)
  -- dexp_13983 aliases dexp_13982
  let {[steps_13765]f32 dexp_13983} = dexp_13982
  let {f32 x_13984} = fsub32(1.0f32, 0.4f32)
  let {f32 x_13985} = fmul32(x_13984, 1.0e-2f32)
  let {bool lifted_1_reduce_arg_13986} =
    lifted_0_reduce_8118(true, true)
  let {f32 lifted_2_reduce_arg_13987, bool lifted_2_reduce_arg_13988} =
    lifted_1_reduce_8119(lifted_1_reduce_arg_13986, 0.0f32)
  let {f32 y_13989} =
    lifted_2_reduce_8124(steps_13765, lifted_2_reduce_arg_13987,
                         lifted_2_reduce_arg_13988, dexp_13983)
  let {f32 CVA_13990} = fmul32(x_13985, y_13989)
  let {f32 CVA_13991} = CVA_13990
  in {CVA_13991, avgexp_13968}
}

entry {f32, *[?0]f32} main (i64 n_13992, i64 n_13993, i64 n_13994,
                            i64 paths_13995, i64 steps_13996,
                            [n_13992]f32 swap_term_13997,
                            [n_13993]i64 payments_13998,
                            [n_13994]f32 notional_13999, f32 a_14000,
                            f32 b_14001, f32 sigma_14002, f32 r0_14003) = {
  let {bool dim_match_14004} = eq_i64(n_13992, n_13993)
  let {cert empty_or_match_cert_14005} =
    assert(dim_match_14004, "function arguments of wrong shape",
           "cva.fut:107:1-181:20")
  -- payments_14006 aliases payments_13998
  let {[n_13992]i64 payments_14006} =
    <empty_or_match_cert_14005>
    reshape((~n_13992), payments_13998)
  let {bool dim_match_14007} = eq_i64(n_13992, n_13994)
  let {cert empty_or_match_cert_14008} =
    assert(dim_match_14007, "function arguments of wrong shape",
           "cva.fut:107:1-181:20")
  -- notional_14009 aliases notional_13999
  let {[n_13992]f32 notional_14009} =
    <empty_or_match_cert_14008>
    reshape((~n_13992), notional_13999)
  let {f32 entry_result_14010, [steps_13996]f32 entry_result_14011} =
    main_7823(n_13992, paths_13995, steps_13996, swap_term_13997,
              payments_14006, notional_14009, a_14000, b_14001, sigma_14002,
              r0_14003)
  in {steps_13996, entry_result_14010, entry_result_14011}
}
