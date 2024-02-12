def functions(type):
    """Add radial functions here to model different profiles."""

    if type == 'power':
        func = "fout[i] = (p[1,i]/( (r/p[2,i])^(-p[5,i]*p[3,i]) + (r/p[2,i])^(-p[5,i]*p[4,i]) )^(1/p[5,i]) )';"

    elif type == 'erf_power':
        func = "fout[i] = p[1,i] * ( erf_in(r, p[2,i], p[3,i]) .* (r/p[2,i])^(p[4,i]) )';"

    elif type == 'erf2_power':
        func = "fout[i] = p[1,i] * ( erf_in(r, p[2,i], p[4,i]) .* erf_out(r, p[5,i], p[6,i]) .* (r/p[2,i])^(p[3,i]) )';"

    elif type == 'gauss':
        func = "fout[i] = p[1,i] * gauss(r, p[2,i], p[3,i])';"

    elif type == 'gauss2':
        func = """
    for (j in 1:N_r) {
        if (r[j] < p[2,i]) {
            fout[i,j] = p[1,i] * exp(-0.5*( (r[j]-p[2,i])/p[3,i] )^2);
        } else {  
            fout[i,j] = p[1,i] * exp(-0.5*( (r[j]-p[2,i])/p[4,i] )^2);
        }
    }
"""

    radial = f"""
matrix radial(data vector r, array [] vector p) {{
    int N_r = size(r);
    int N_p = size(p[1]);
    matrix[N_p,N_r] fout;
    for (i in 1:N_p) {{
        {func}
    }}
    return fout;
}}
    """

    return f"""
functions {{
    array[] vector interp_1d_linear(array[] vector y, data array[] real x,
                                    array[] real x_out) {{
      int left = 1;
      int right = 1;
      real w = 1.0;
      int N_in = size(x);
      int N_out = size(x_out);
      int D = size(y[1]);
      array[N_out] vector[D] y_out;
      for (j in 1 : N_out) {{
        while (x[right] < x_out[j]) {{
          right = right + 1;
        }}
        while (x[left + 1] < x_out[j]) {{
          left = left + 1;
        }}
        w = (x[right] - x_out[j]) / (x[right] - x[left]);
        y_out[j] = w * y[left] + (1 - w) * y[right];
      }}
      return y_out;
    }}
    
    vector gauss(data vector r, real r0, real sig) {{
        return exp(-0.5 * square((r-r0)/sig) );
    }}
    vector erf_in(data vector r, real r0, real sigi) {{
        return 0.5 * erfc( (r0-r)/(1.4142135624*sigi*r0) );
    }}
    vector erf_out(data vector r, real r0, real sigo) {{
        return 0.5 * erfc( (r-r0)/(1.4142135624*sigo*r0) );
    }}
    
    {radial}
}}
"""


def data(pn, star=False, bg=False, pt=False):

    mul_str = f'{pn[0]}_mul'
    z_str = f'{pn[0]}_0'
    for p in pn[1:]:
        mul_str += f', {p}_mul'
        z_str += f', {p}_0'

    star_str = ''
    if star:
        star_str = 'real star_0, star_mul;'

    bg_str = '// no bg'
    if bg:
        bg_str = """
    vector[nbg] bgx_0, bgy_0,  bgn_0, bgr_0, bgpa_0, bgi_0;
    real bgx_mul, bgy_mul, bgn_mul, bgr_mul, bgpa_mul, bgi_mul;
    """

    pt_str = '// no pt'
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
    vector[nr] zh_0;
    real zh_mul;
    real {mul_str};
    vector[nr] {z_str};
    int nhpt;
    matrix[nhpt,nhpt] Ykm;
    vector[nhpt] Rnk, Qnk;
    real hnorm;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""


transformed_data = """
transformed data {
    // constants
    real arcsec = pi()/180/3600;
    real arcsec2pi = arcsec * 2*pi();
    // data, convert u,v to sky x,y
    vector[nvis] u_ = u * arcsec2pi;
    vector[nvis] v_ = v * arcsec2pi;
    vector[nhpt+1] Qnk_;
    Qnk_[1] = 0;
    for (i in 1:nhpt) {
        Qnk_[i+1] = Qnk[i];
    }

}
"""

def parameters(pn, pl, pu, star=False, bg=False, pt=False, inc_lim=False, pa_lim=False, zh_lim=True, nbg_lim=True):
    inc = '<lower=0, upper=inc_mul*pi()/2>' if inc_lim else ''
    pa = '<lower=0, upper=inc_mul*pi()>' if pa_lim else ''
    zh = '<lower=0>' if zh_lim else ''
    nbg = '<lower=0>' if nbg_lim else ''

    p_str = ''
    for p, l, u in zip(pn, pl, pu):
        p_str += f'    vector'
        if l is not None or u is not None:
            p_str += '<'
            if l is not None:
                p_str += f'lower={l} '
                if u is not None:
                    p_str += ','
            if u is not None:
                p_str += f'upper={u}'
            p_str += '>'
        p_str += f'[nr] {p};\n'

    star_str = '// no star'
    if star:
        star_str = 'real<lower=0> star;'

    bg_str = '// no bg'
    if bg:
        bg_str = f"""
    vector[nbg] bgx, bgy, bgpa;
    vector{nbg}[nbg] bgi, bgr, bgn;
    """

    pt_str = '// no pt'
    if pt:
        pt_str = f"""
    vector[npt] ptx, pty;
    vector{nbg}[npt] ptn;
    """

    return f"""
parameters {{
    real dra;
    real ddec;
    real{pa} pa;
    real{inc} inc;
    {p_str}
    vector{zh}[nr] zh;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""


def transformed_parameters(pn, star=False, bg=False, pt=False):

    p_str = ''
    for p in pn:
        p_str += f'    vector[nr] {p}_ = {p}/{p}_mul;\n'

    star_str = '// no star'
    if star:
        star_str = 'real star_ = star/star_mul;'

    bg_str = '// no bg'
    if bg:
        bg_str = """
    vector[nbg] bgx_ = bgx/bgx_mul;
    vector[nbg] bgy_ = bgy/bgy_mul;
    vector[nbg] bgn_ = bgn/bgn_mul;
    vector[nbg] bgr_ = bgr/bgr_mul;
    vector[nbg] bgpa_ = bgpa/bgpa_mul;
    vector[nbg] bgi_ = bgi/bgi_mul;
    """

    pt_str = '// no pt'
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
{p_str}
    vector[nr] zh_ = zh/zh_mul;
    {star_str}
    {bg_str}
    {pt_str}
}}
"""


def model_core(pn, star=False, bg=False, pt=False, gq='vis'):

    p_str = f'{pn[0]}_'
    np_str = 1
    for p in pn[1:]:
        p_str += f', {p}_'
        np_str += 1

    star_str = 'mod'
    if star:
        star_str = '(mod + star_)'

    bg_str = '// no bg'
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

    pt_str = '// no pt'
    if pt:
        pt_str = """
            if (npt>0) {
                matrix[npt, nvis] ruv3 = ptx_ * u_' + pty_ * v_';
                vismod_re += (ptn_' * cos(ruv3))';
                vismod_im += (ptn_' * sin(ruv3))';
            }
"""

    vis_str = """
    vector[nvis] vismod_re;
    vector[nvis] vismod_im;
"""

    f_str = """
    matrix[nhpt,nr] vnk, f;
"""

    if gq == 'vis' or gq is False:
        vis = True
        prof = False
    elif gq == 'prof':
        vis = False
        prof = True

    return f"""
    {vis_str if vis else ''}
    {f_str if prof else ''}

    {{
        {vis_str if not vis else ''}
        {f_str if not prof else ''}
        vector[nvis] mod;
        vector[nvis] ruv;
        array[nvis] int ruvi;
        vector[nvis] ruv2;
        real cos_pa = cos(pa_);
        real sin_pa = sin(pa_);
        vector[nvis] vsin_pa = v_ * sin_pa;
        vector[nvis] ucos_pa = u_ * cos_pa;
        vector[nvis] urot;
        vector[nvis] vrot;
        array[nhpt+1] vector[nr] vnk_;
        matrix[nvis,nr] mod2d;
        array[nvis] vector[nr] mod2d_;
        array[{np_str}] vector[nr] pars = {{ {p_str} }};
        
        profile("rotation"){{
            urot = ucos_pa - vsin_pa;
            vrot = u_*sin_pa + v_*cos_pa;
        }}
        
        profile("sqrt"){{
            ruv2 = square(urot*cos(inc_)) + square(vrot);
            ruv = sqrt(ruv2);
            ruvi = sort_indices_asc(ruv);
            ruv = sort_asc(ruv);
        }}

        profile("radial"){{
            f = radial(Rnk, pars)';
        }}

        profile("hankel"){{
            for (i in 1:nr) {{
                vnk[:,i] = hnorm * Ykm * f[:,i];
            }}
            vnk_[1] = vnk[1]';
            for (i in 1:nhpt) {{
                vnk_[i+1] = vnk[i]';
            }}
        }}

        profile("interp"){{
            mod2d_ = interp_1d_linear(vnk_, to_array_1d(Qnk_), to_array_1d(ruv/arcsec2pi));
            for (i in 1:nvis) {{
                mod2d[ruvi[i]] = mod2d_[i]';
            }}
        }}

        profile("vertical"){{
            matrix[nr, nvis] rz = (zh_ .* r_) * sin(inc_) * urot';
            mod = rows_dot_product(mod2d, exp(-0.5*(square(rz)))');
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


def model_lnprob(pn, star=False, bg=False, pt=False, z_prior=None):

    zpr_str = 5
    if z_prior is not None:
        zpr_str = z_prior

    p_str = ''
    for p in pn:
        p_str += f'    target += normal_lpdf({p} | {p}_0, 5);\n'

    star_str = '// no star'
    if star:
        star_str = 'target += normal_lpdf(star | star_0, 5);'

    bg_str = '// no bg'
    if bg:
        bg_str = """
    target += normal_lpdf(bgx | bgx_0, 5);
    target += normal_lpdf(bgy | bgy_0, 5);
    target += normal_lpdf(bgn | bgn_0, 5);
    target += normal_lpdf(bgr | bgr_0, 5);
    target += normal_lpdf(bgpa | bgpa_0, 5);
    target += normal_lpdf(bgi | bgi_0, 5);
    """

    pt_str = '// no pt'
    if pt:
        pt_str = """
    target += normal_lpdf(ptx | ptx_0, 5);
    target += normal_lpdf(pty | pty_0, 5);
    target += normal_lpdf(ptn | ptn_0, 5);
    """

    return f"""
    // priors
{p_str}
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


def get_code(type, star=False, bg=False, pt=False, gq=False,
             inc_lim=False, pa_lim=False, z_prior=None):

    if type == 'power':
        pn = ['norm', 'r', 'ai', 'ao', 'gam']
        pl = [0,       0,   0,    None, 0]
        pu = [None,  None,  None, 0,    None]
    elif type == 'erf_power':
        pn = ['norm', 'r', 'sigi', 'ao']
        pl = [0,       0,   0,      None]
        pu = [None,  None,  None,   0]
    elif type == 'erf2_power':
        pn = ['norm', 'ri', 'ai', 'sigi', 'ro', 'sigo']
        pl = [0,       0,    0,    0,      None, 0]
        pu = [None,  None,   None, None,   None, None]
    elif type == 'gauss':
        pn = ['norm', 'r', 'dr']
        pl = [0,       0,   0]
        pu = [None,  None,  None]
    elif type == 'gauss2':
        pn = ['norm', 'r', 'dri', 'dro']
        pl = [0,       0,   0,     0]
        pu = [None,  None,  None,  None]
    else:
        exit(f'need to add function: {type}')

    model = "model {\n" + model_core(pn, star=star, bg=bg, pt=pt) + \
            model_lnprob(pn, star=star, bg=bg, pt=pt, z_prior=z_prior)

    generated_quantities = "generated quantities {" + model_core(pn, star=star, bg=bg, pt=pt, gq=gq) + "\n}"

    code = functions(type) + data(pn, star=star, bg=bg, pt=pt) + transformed_data + \
           parameters(pn, pl, pu, star=star, bg=bg, pt=pt, inc_lim=inc_lim, pa_lim=pa_lim) + \
           transformed_parameters(pn, star=star, bg=bg, pt=pt)
    if gq is not False:
        return code + generated_quantities
    else:
        return code + model
