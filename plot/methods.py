perts = [
    "mlef", "mlef_nest", "mlef_nestc", \
    "envar", "envar_nest", "envar_nestc",\
    "etkf", "po", "srf", "eakf", "letkf", "kf", "var","var_nest","var_nestc",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"
    ]
linecolor = {
    "mlef":'tab:blue',
    "mlef_nest":'tab:purple',
    "mlef_nestc":'tab:cyan',
    "envar":'tab:orange',
    "envar_nest":'tab:green',
    "envar_nestc":"lime",
    "etkf":'tab:green', 
    "po":'tab:red',
    "srf":"tab:pink", 
    "eakf":"tab:gray", 
    "letkf":"tab:purple", 
    "kf":"tab:cyan", 
    "var":"tab:olive",
    "var_nest":"tab:brown",
    "var_nestc":"gold",
    "mlefcw":"tab:green",
    "mlefy":"tab:orange",
    "mlefbe":"tab:red",
    "mlefbm":"tab:pink"
    }infltype = {
    -3:r'$g(\gamma)=\rho\gamma$',
    -2:'adap-pre-mi',
    -1:'pre-mi',
     0:'post-mi', 
     1:'add', 
     2:'rtpp', 
     3:'rtps',
     4:r'mult,1/$\lambda_\mathrm{ave}$',
     5:r'mult,$\lambda_\mathrm{ave}$'
    }
inflcolor = {
    -3:'magenta',
    -2:'tab:cyan',
    -1:'tab:blue', 
     0:'tab:orange', 
     1:'tab:green', 
     2:'tab:purple', 
     3:'r', 
     4:'b',
     5:'g',
    }
inflmarkers = {
    -3:'v',
    -2:'',
    -1:'',
    0:'',
    1:'',
    2:'',
    3:'o',
    4:'s',
    5:'*',
}
iinflist = list(infltype.keys())