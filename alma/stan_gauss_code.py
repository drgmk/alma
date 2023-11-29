functions = """
functions {
    vector bessel_annulus(vector norm_, vector r_, vector dr_, vector zh_,
                          vector ruv, real sin_inc_, vector urot, 
                          int nr, int nvis) {
        vector[nr] ri = r_ - dr_/2;
        vector[nr] ro = r_ + dr_/2;
        vector[nr] dr_2 = square(ri ./ ro);
        vector[nr] norm_pos = norm_ ./ (1.0-dr_2);
        vector[nr] norm_neg = norm_ .* dr_2 ./ (1.0-dr_2);
        matrix[nr, nvis] rz = (zh_ .* r_) * sin_inc_ * urot';
        vector[nvis] cr_pos = (norm_pos' * (bessel_first_kind(1, ro * ruv') ./ (ro/2 * ruv') .* exp(-0.5*square(rz))))';
        vector[nvis] cr_neg = (norm_neg' * (bessel_first_kind(1, ri * ruv') ./ (ri/2 * ruv') .* exp(-0.5*square(rz))))';
        return cr_pos - cr_neg;
    }
}
"""

def data(star=False, bg=False, pt=False):

    star_str = ''
    if star:
        star_str = 'real star_0, star_mul;'

    bg_str = ''
    if bg:
        bg_str = """
    vector[nbg] bgx_0, bgy_0,  bgn_0, bgr_0, bgpa_0, bgi_0;
    real bgx_mul, bgy_mul, bgn_mul, bgr_mul, bgpa_mul, bgi_mul;
    """

    pt_str = ''
    if pt:
        pt_str = """
    vector[npt] ptx_0, pty_0,  ptn_0;
    real ptx_mul, pty_mul, ptn_mul;
    """

    return f"""
data {{
    int nvis, nbg, npt, nr;
    vector[nvis] u;
    vector[nvis] v;
    array[nvis] real re;
    array[nvis] real im;
    vector[nvis] sigma;
    real pa_0, pa_mul, inc_0, inc_mul;
    real dra_0, dra_mul, ddec_0, ddec_mul;
    real norm_mul, r_mul, dr_mul, zh_mul;
    vector[nr] norm_0, r_0, dr_0, zh_0;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""

transformed_data = """
transformed data {
    // constants
    real arcsec2pi = pi()/180/3600 * 2*pi();
    // data, convert u,v to sky x,y
    vector[nvis] u_ = u * arcsec2pi;
    vector[nvis] v_ = v * arcsec2pi;
}
"""

def parameters(star=False, bg=False, pt=False, inc_lim=False, r_lim=False, dr_lim=False, zh_lim=True, nbg_lim=True):
    inc = '<lower=0, upper=inc_mul*pi()/2>' if inc_lim else ''
    r = '<lower=0>' if r_lim else ''
    dr = '<lower=0>' if dr_lim else ''
    zh = '<lower=0>' if zh_lim else ''
    nbg = '<lower=0>' if nbg_lim else ''

    star_str = '// no star'
    if star:
        star_str = 'real<lower=0> star;'

    bg_str = ''
    if bg:
        bg_str = f"""
    vector[nbg] bgx, bgy, bgpa;
    vector{nbg}[nbg] bgi, bgr, bgn;
    """

    pt_str = ''
    if pt:
        pt_str = f"""
    vector[npt] ptx, pty;
    vector{nbg}[npt] ptn;
    """

    return f"""
parameters {{
    real dra;
    real ddec;
    real pa;
    real{inc} inc;
    vector{r}[nr] r;
    vector[nr] norm;
    vector{dr}[nr] dr;
    vector{zh}[nr] zh;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""

def transformed_parameters(star=False, bg=False, pt=False):

    star_str = '// no star'
    if star:
        star_str = 'real star_ = star/star_mul;'

    bg_str = ''
    if bg:
        bg_str = """
    vector[nbg] bgx_ = bgx/bgx_mul;
    vector[nbg] bgy_ = bgy/bgy_mul;
    vector[nbg] bgn_ = bgn/bgn_mul;
    vector[nbg] bgr_ = bgr/bgr_mul;
    vector[nbg] bgpa_ = bgpa/bgpa_mul;
    vector[nbg] bgi_ = bgi/bgi_mul;
    """

    pt_str = ''
    if pt:
        pt_str = """
    vector[npt] ptx_ = ptx/ptx_mul;
    vector[npt] pty_ = pty/pty_mul;
    vector[npt] ptn_ = ptn/ptn_mul;
    """

    return f"""
transformed parameters {{
    real dra_ = dra/dra_mul;
    real ddec_ = ddec/ddec_mul;
    real pa_ = pa/pa_mul;
    real inc_ = inc/inc_mul;
    vector[nr] norm_ = norm/norm_mul;
    vector[nr] r_ = r/r_mul;
    vector[nr] dr_ = dr/dr_mul;
    vector[nr] zh_ = zh/zh_mul;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""

def model_core(star=False, bg=False, pt=False):

    star_str = 'mod'
    if star:
        star_str = '(mod + star_)'

    bg_str = ''
    if bg:
        bg_str = """
            if (nbg>0) {
                for (i in 1:nbg) {
                    urot = cos(bgpa_[i]) * u_ - sin(bgpa_[i]) * v_;
                    vrot = sin(bgpa_[i]) * u_ + cos(bgpa_[i]) * v_;
                    ruv2 = square(urot*cos(bgi_[i])) + square(vrot);
                    mod = bgn_[i] * exp(-0.5*square(bgr_[i])*ruv2);
                    ruv = bgx_[i] * u_ + bgy_[i] * v_;
                    vismod_re += mod .* cos(ruv);
                    vismod_im += mod .* sin(ruv);
                }
            }
"""

    pt_str = ''
    if pt:
        pt_str = """
            if (npt>0) {
                matrix[npt, nvis] ruv3 = ptx_ * u_' + pty_ * v_';
                vismod_re += (ptn_' * cos(ruv3))';
                vismod_im += (ptn_' * sin(ruv3))';
            }
"""

    return f"""
    vector[nvis] vismod_re;
    vector[nvis] vismod_im;

    {{
        vector[nvis] mod;
        vector[nvis] ruv;
        vector[nvis] ruv2;
        real cos_pa = cos(pa_);
        real sin_pa = sin(pa_);
        vector[nvis] vsin_pa = v_ * sin_pa;
        vector[nvis] ucos_pa = u_ * cos_pa;
        vector[nvis] urot;
        vector[nvis] vrot;
        
        profile("rotation"){{
            urot = ucos_pa - vsin_pa;
            vrot = u_*sin_pa + v_*cos_pa;
        }}
        
        profile("sqrt"){{
            ruv2 = square(urot*cos(inc_)) + square(vrot);
            ruv = sqrt(ruv2);
        }}
        
        profile("bessel"){{
            matrix[nr, nvis] rz = (zh_ .* r_) * sin(inc_) * urot';
            mod = (norm_' * (bessel_first_kind(0, r_ * ruv') .* exp(-0.5*(square(dr_)*ruv2' + square(rz)))))';
//            mod = bessel_annulus(norm_, r_, dr_, zh_, ruv, sin(inc_), urot, nr, nvis);
        }}
        
        profile("translate"){{
            ruv = u_*dra_ + v_*ddec_;
            vismod_re = {star_str} .* cos(ruv);
            vismod_im = mod .* sin(ruv);
        }}
        
        profile("background"){{
            {bg_str}
            {pt_str}
        }}
    }}

"""

def model_lnprob(star=False, bg=False, pt=False, z_prior=None):

    zpr_str = 5
    if z_prior is not None:
        zpr_str = z_prior

    star_str = '// no star'
    if star:
        star_str = 'target += normal_lpdf(star | star_0, 5);'

    bg_str = ''
    if bg:
        bg_str = """
    target += normal_lpdf(bgx | bgx_0, 5);
    target += normal_lpdf(bgy | bgy_0, 5);
    target += normal_lpdf(bgn | bgn_0, 5);
    target += normal_lpdf(bgr | bgr_0, 5);
    target += normal_lpdf(bgpa | bgpa_0, 5);
    target += normal_lpdf(bgi | bgi_0, 5);
    """

    pt_str = ''
    if pt:
        pt_str = """
    target += normal_lpdf(ptx | ptx_0, 5);
    target += normal_lpdf(pty | pty_0, 5);
    target += normal_lpdf(ptn | ptn_0, 5);
    """

    return f"""
    // priors
    target += normal_lpdf(norm | norm_0, 5);
    target += normal_lpdf(r | r_0, 5);
    target += normal_lpdf(abs(dr) | dr_0, 5);
    target += normal_lpdf(abs(zh) | zh_0, {zpr_str});
    target += normal_lpdf(pa | pa_0, 5);
    target += normal_lpdf(inc | inc_0, 5);
    target += normal_lpdf(dra | dra_0, 5);
    target += normal_lpdf(ddec | ddec_0, 5);
    {star_str}
    {bg_str}
    {pt_str}
    // log probability
    profile("lnprob"){{
        target += normal_lupdf(vismod_re | re, sigma);
        target += normal_lupdf(vismod_im | im, sigma);
    }}
}}
"""


def get_code(star=False, bg=False, pt=False, gq=False,
             r_lim=False, dr_lim=False, inc_lim=False, z_prior=None):

    model = "model {\n" + model_core(star=star, bg=bg, pt=pt) + \
            model_lnprob(star=star, bg=bg, pt=pt, z_prior=z_prior)

    generated_quantities = "generated quantities {" + model_core(star=star, bg=bg, pt=pt) + "\n}"

    code = functions + data(star=star, bg=bg, pt=pt) + transformed_data + \
           parameters(star=star, bg=bg, pt=pt, inc_lim=inc_lim, r_lim=r_lim, dr_lim=dr_lim) + \
           transformed_parameters(star=star, bg=bg, pt=pt)
    if gq:
        return code + generated_quantities
    else:
        return code + model

"""
// unused code

/*
vector translate_1(vector dxy, vector mod, array[] real uv, array[] int tmp) {
    real rot = -uv[1]*dxy[1] + uv[2]*dxy[2];
vector[2] mod_re_im = [mod[1] * cos(rot), mod[1] * sin(rot)]';
return mod_re_im;
}
vector translate_n(vector dxy, vector mod, array[] real uv, array[] int nz) {
real rot0 = -uv[1]*dxy[1] + uv[2]*dxy[2];
vector[2] mod_re_im = [mod[1] * cos(rot0), mod[1] * sin(rot0)]';
for (n in 1:nz[1]) {
    real rot_ = dxy[n+2] * mod[2] + rot0;
real tmp1 = dxy[n+2+nz[1]] * mod[1];
mod_re_im[1] += tmp1 * cos(rot_);
mod_re_im[2] += tmp1 * sin(rot_);
}
return mod_re_im;
} */

//  array[nvis,1] int tmp2;
//  for (n in 1:nvis) { tmp2[n,1] = nz; }
//  array[nvis,2] real uv;
//  uv[:,1] = to_array_1d(u_);
//  uv[:,2] = to_array_1d(v_);

/*        profile("crescent"){
            if (ncr>0) {
                real ri = r_[1]-dr_[1]/4;
                real ro = r_[1]+dr_[1]/4;
                real dr_2 = square(ri/ro);
                real norm_pos = crn_/(1.0-dr_2);
                real norm_neg = crn_*dr_2/(1.0-dr_2);
                vector[nvis] cr_pos = norm_pos * bessel_first_kind(1, ro * ruv) ./ (ro * ruv);
                vector[nvis] cr_neg = norm_neg * bessel_first_kind(1, ri * ruv) ./ (ri * ruv);
                vismod_re += cr_pos;
                ruv = dr_[1]/2*cos(crpa_) * urot*cos(inc_) - dr_[1]/2*sin(crpa_) * vrot;
                vismod_re -= cr_neg .* cos(ruv);
                vismod_im -= cr_neg .* sin(ruv);
            }            
        } */
        
/*                matrix[nbg, nvis] urot2 = (cos(bgi_).*cos(bgpa_)) * u_' - (cos(bgi_).*sin(bgpa_)) * v_';
                matrix[nbg, nvis] vrot2 = sin(bgpa_) * u_' + cos(bgpa_) * v_';
                matrix[nbg, nvis] ruv3 = square(urot2 + vrot2);
                matrix[nbg, nvis] mod2;
                for (i in 1:nvis) {
                    mod2[:, i] = exp(-0.5*square(bgr_) .* ruv3[:, i]);
                }
                ruv3 = bgx_ * u_' + bgy_ * v_';
                vismod_re += (bgn_' * (mod2 .* cos(ruv3)))';
                vismod_im += (bgn_' * (mod2 .* sin(ruv3)))'; */
                
/*        profile("bessel"){
            mod = bessel_first_kind(0, ruv*r_) .* exp(-0.5*square(ruv*dr_));
        }
        
        profile("vertical"){
            ruv = zh_ * r_ * sin(inc_) * (ucos_pa + vsin_pa);
            mod = mod .* exp(-0.5*square(ruv));
        } */

/*  {
profile("translate_map"){
    array[nvis] vector[1] mod_arr;
mod_arr[:,1] = to_array_1d(mod);
vector[2] shifts = [dra_, ddec_]';
vector[2*nvis] vismod_tmp = map_rect(translate_1, shifts, mod_arr, uv, tmp2);
for (n in 1:nvis) {
    vismod_re[n] = vismod_tmp[1+(n-1)*2];
vismod_im[n] = vismod_tmp[2+(n-1)*2];
}
}
} */

/*    profile("layers"){
    if (nz > 0) {
        vector[nvis] tmp = zh_ * r_ * sin(inc_) * (ucos_pa + vsin_pa);
        for (n in 1:nz) {
            vector[nvis] rot_ = dz[n] * tmp + rot;
            vector[nvis] tmp1 = zwt_[n] * mod;
            vismod_re += tmp1 .* cos(rot_);
            vismod_im += tmp1 .* sin(rot_);
        }
    } else {}
    } */
    
        profile("layers"){
        if (nz > 0) {
            vector[nvis] mod_tmp;
            for (n in 1:nvis) { mod_tmp[n] = 0; }
            rot = zh_ * r_ * sin(inc_) * (ucos_pa + vsin_pa);
            for (n in 1:nz) {
                mod_tmp += zwt_[n] * mod .* cos(dz[n] * rot);
            } 
            mod += mod_tmp;
        } else {}
        }


/*  {
profile("layers_map"){
    array[nvis] vector[2] mod_tmp;
mod_tmp[:,1] = to_array_1d(mod);
mod_tmp[:,2] = to_array_1d(zh_ * r_ * sin(inc_) * (ucos_pa + vsin_pa));
vector[2*(nz+1)] shifts;
shifts[1:2] = [dra_, ddec_]';
for (n in 1:nz) {
    shifts[n+2] = dz[n];
shifts[n+2+nz] = zwt_[n];
}
vector[2*nvis] vismod_tmp = map_rect(translate_n, shifts, mod_tmp, uv, tmp2);
for (n in 1:nvis) {
    vismod_re[n] = vismod_tmp[1+(n-1)*2];
vismod_im[n] = vismod_tmp[2+(n-1)*2];
}
}
} */

//        matrix[nvis,nz] rot_ = tmp * dz';
                                         //        matrix[nvis,nz] tmp1 = mod * zwt_';
                                                                                    //        for (n in 1:nz) {
//          vismod_re += tmp1[:,n] .* cos(rot_[:,n]+rot);
//          vismod_im += tmp1[:,n] .* sin(rot_[:,n]+rot);
//        }
"""