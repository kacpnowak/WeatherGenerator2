import yaml

# GLORYS levels (31 levels down to ~454m)
depths = [0.494025, 1.541375, 2.645669, 3.819495, 5.078224, 6.440614, 7.92956, 9.572997, 11.405, 13.46714, 15.81007, 18.49556, 21.59882, 25.21141, 29.44473, 34.43415, 40.34405, 47.37369, 55.76429, 65.80727, 77.85385, 92.32607, 109.7293, 130.666, 155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.213, 453.9377]
depth_strs = [f"{d:.6g}m" for d in depths]

glorys_ocean_vars = ['zos', 'mlotst'] # OceanBench parameter + mixed layer thickness
for prefix in ['thetao', 'so', 'uo', 'vo']: # OceanBench parameters: temp, salt, u, v
    glorys_ocean_vars.extend([f"{prefix}_{d}" for d in depth_strs])

glorys_atmos_vars = ['sotemair', 'sod2m', 'sowinu10', 'sowinv10', 'somslpre', 'sowaprec', 'sosudosw', 'sosudolw']

glorys_stream = {
    'GLORYS_OCEAN': {
        'type': 'mesh',
        'stream_id': 2136,
        'filenames': ['/e/data1/climateai/hclimrep/data/glorys/glorys_full_mesh.parq'],
        'source': glorys_ocean_vars,
        'target': None,
        'sampling_mode': 'global_sparse',
        'sample_points': 4096,
        'loss_weight': 1.0,
        'location_weight': 'cosine_latitude',
        'masking_rate': 0.6,
        'masking_rate_none': 0.05,
        'token_size': 32,
        'tokenize_spacetime': True,
        'embed': {
            'net': 'transformer',
            'num_tokens': 1,
            'num_heads': 8,
            'dim_embed': 256,
            'num_blocks': 2
        }
    },
    'GLORYS_ATMOS': {
        'type': 'grid',
        'stream_id': 2137,
        'filenames': ['/e/data1/climateai/hclimrep/data/glorys_forcings/ifs/'],
        'source': glorys_atmos_vars,
        'target': None,
        'loss_weight': 1.0,
        'masking_rate': 0.6,
        'masking_rate_none': 0.05,
        'token_size': 128,
        'embed': {
            'net': 'transformer',
            'num_tokens': 1,
            'num_heads': 8,
            'dim_embed': 256,
            'num_blocks': 2
        }
    }
}

# FESOM levels (20 levels down to 450m)
fesom_ocean_depths = [2, 7, 15, 25, 35, 45, 55, 65, 75, 85, 95, 107, 125, 147, 175, 210, 255, 310, 375, 450]

fesom_node_vars = ['ssh', 'MLD2'] # OceanBench parameter + mixed layer depth
for prefix in ['temp', 'salt']: # OceanBench parameters: temp, salt
    fesom_node_vars.extend([f"{prefix}_{d}" for d in fesom_ocean_depths])

fesom_elem_vars = []
for prefix in ['u', 'v']: # OceanBench parameters: u, v velocities
    fesom_elem_vars.extend([f"{prefix}_{d}" for d in fesom_ocean_depths])

# Only surface atmospheric variables (1-1 with GLORYS atmos)
ifs_atmos_vars = ['2t', '2d', '10u', '10v', 'msl', 'cp', 'lsp', 'tsr', 'tsrc']

ifs_fesom_stream = {
    'FESOM_NODE': {
        'type': 'fesom',
        'stream_id': 2138,
        'filenames': ['ocean_node'],
        'source': fesom_node_vars,
        'target': None,
        'loss_weight': 1.0,
        'masking_rate': 0.6,
        'masking_rate_none': 0.05,
        'token_size': 128,
        'embed': {
            'net': 'transformer',
            'num_tokens': 1,
            'num_heads': 8,
            'dim_embed': 256,
            'num_blocks': 2
        }
    },
    'FESOM_ELEM': {
        'type': 'fesom',
        'stream_id': 2140,
        'filenames': ['ocean_elem'],
        'source': fesom_elem_vars,
        'target': None,
        'loss_weight': 1.0,
        'masking_rate': 0.6,
        'masking_rate_none': 0.05,
        'token_size': 128,
        'embed': {
            'net': 'transformer',
            'num_tokens': 1,
            'num_heads': 8,
            'dim_embed': 256,
            'num_blocks': 2
        }
    },
    'IFS_ATMOS': {
        'type': 'fesom',
        'stream_id': 2139,
        'filenames': ['atmos_all'],
        'source': ifs_atmos_vars,
        'target': None,
        'loss_weight': 1.0,
        'masking_rate': 0.6,
        'masking_rate_none': 0.05,
        'token_size': 128,
        'embed': {
            'net': 'transformer',
            'num_tokens': 1,
            'num_heads': 8,
            'dim_embed': 256,
            'num_blocks': 2
        }
    }
}

with open('/e/scratch/hclimrep/nowak2/WeatherGenerator2/config/streams/glorys_coupled.yml', 'w') as f:
    yaml.dump(glorys_stream, f, sort_keys=False)

with open('/e/scratch/hclimrep/nowak2/WeatherGenerator2/config/streams/ifs_fesom_coupled.yml', 'w') as f:
    yaml.dump(ifs_fesom_stream, f, sort_keys=False)

print("Streams successfully written to match OceanBench requirements.")
