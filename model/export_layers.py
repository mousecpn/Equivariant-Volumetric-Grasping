import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torchvision.ops as ops
from utils.common import normalize_coordinate, normalize_3d_coordinate
from utils.transform import quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d


class ExportResBlock2d(nn.Module):
    def __init__(self, ):
        super(ExportResBlock2d, self).__init__()
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.non_linearity = None
    
    def forward(self, input):
        out = self.conv1(input)
        residual = out
        
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        out = self.non_linearity(out)
        
        return out


class ExportDownConv2d(nn.Module):
    def __init__(self, ):
        super(ExportDownConv2d, self).__init__()
        self.pooling = None        
        self.conv = None
        self.pool = None
    
    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv(x)
        return x

class ExportUpConv2d(nn.Module):
    def __init__(self, ):
        super(ExportUpConv2d, self).__init__()
        self.merge_mode = None
        self.conv = None
        self.upconv = None
    
    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down
        elif self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), dim=1)
        x = self.conv(x)
        return x

class ExportCyclicUnet2d(nn.Module):
    def __init__(self, ):
        super(ExportCyclicUnet2d, self).__init__()        
        self.encoders = None
        self.decoders = None        
        self.final_conv = None
    
    def forward(self, x):
        encoders_features = []
        
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            
        
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        
        return x

class ExportJointTriUNet(nn.Module):
    def __init__(self, ):
        super(ExportJointTriUNet, self).__init__()
        self.lifting = None
        self.encoders_xy = None
        self.down_convs = None
        self.side2xy_encoders = None
        self.decoders_xy = None
        self.up_convs = None
        self.side2xy_decoders = None
        self.final_conv_xy = None
        self.conv_final = None
        self.lifting_dim = None
        self.reso_plane = None
        self.padding = None
        self.plane_type = None

    def voxel2repr(self, voxel):
        shape_len = len(voxel.shape)
        if shape_len == 5:
            repr = rearrange(voxel, "b c h w d -> b c d w h")
        elif shape_len == 4:
            repr = rearrange(voxel, "b h w d -> b d w h")
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        return repr
    
    def repr2voxel(self, repr):
        shape_len = len(repr.shape)
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        voxel = rearrange(repr, "b c d w h -> b c h w d")
        return voxel
    
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x = self.voxel2repr(x)

        # Acquire voxel-wise feature
        # c = self.act(self.lifting(x))
        c = self.lifting(x)
        # inv_voxel = self.inv_map(c)
        c = self.repr2voxel(c)
        batch_size, dim = c.shape[:2]
        c = c.reshape(batch_size, dim, -1)
        c = c.permute(0, 2, 1)
        
        fea = {}
        
        fea_plane = c.permute(0,2,1).reshape(-1, self.lifting_dim, self.reso_plane, self.reso_plane, self.reso_plane)
        # fea['xy'] = fea_plane.mean(-1)
        # fea['xz'] = fea_plane.mean(-2)
        # fea['yz']  = fea_plane.mean(-3)
        tri_feature = torch.stack((fea_plane.mean(-1),fea_plane.mean(-2),fea_plane.mean(-3)), dim=0)
        
        encoders_features = []
        
        for i in range(len(self.encoders_xy)):
            xy_feature = self.encoders_xy[i](tri_feature[0])
            xz_feature = self.down_convs[i](tri_feature[1])
            yz_feature = self.down_convs[i](tri_feature[2])
            
            add_xy = (xz_feature.mean(-1)[:,:,:,None] + yz_feature.mean(-1)[:,:,None,:])
            xy_feature = torch.cat((xy_feature, add_xy), dim=1)
            xy_feature = self.side2xy_encoders[i](xy_feature)
            
            tri_feature = torch.stack((xy_feature, xz_feature, yz_feature), dim=0)
            
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, tri_feature)
            
        
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for i, (decoder_xy, encoder_features) in enumerate(zip(self.decoders_xy, encoders_features)):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            xy_feature = decoder_xy(encoder_features[0], tri_feature[0])
            
            xz_feature = self.up_convs[i](encoder_features[1], tri_feature[1])
            yz_feature = self.up_convs[i](encoder_features[2], tri_feature[2])
            
            add_xy = (xz_feature.mean(-1)[:,:,:,None] + yz_feature.mean(-1)[:,:,None,:])
            xy_feature = torch.cat((xy_feature, add_xy), dim=1)
            xy_feature = self.side2xy_decoders[i](xy_feature)
            
            tri_feature = torch.stack((xy_feature, xz_feature,yz_feature), dim=0)

        fea['xy'] = self.final_conv_xy(tri_feature[0])
        fea['xz'] = self.conv_final(tri_feature[1])
        fea['yz'] = self.conv_final(tri_feature[2])

        fea['xy'] = fea['xy'].permute(0,1,3,2)
        fea['xz'] = fea['xz'].permute(0,1,3,2) 
        fea['yz'] = fea['yz'].permute(0,1,3,2)   # from pixel coord to real coord
        
        return fea

class ExportEquiDeformConv(nn.Module):
    def __init__(self, ):
        super(ExportEquiDeformConv, self).__init__()
        self.conv_offset = None
        self.conv = None
        self.mask1 = None
        self.mask2 = None
        self.base_offset = None

    def forward(self, x):
        offset = self.conv_offset(x)

        bs, _, h, w = offset.shape
        scale1 = F.relu(offset[:, 2:3]) + 1
        scale2 = F.relu(offset[:, 3:4]) + 1
        # scale1 = F.elu(offset[:, 2:3]) + 1
        # scale2 = F.elu(offset[:, 3:4]) + 1
        shift = offset[:, :2].repeat(1, 9, 1, 1)
        ### debug ###
        # basic_rot = unnormalized_vector_to_rot_matrix(offset[:, 4:]) 
        
        # basic_rot = offset[:, 4:].clip(-1,1)*(torch.pi/self.N)
        # basic_rot = theta_to_rot_matrix(basic_rot)
        
        
        _filter, _bias = self.conv.weight, self.conv.bias
        offset = torch.zeros_like(shift)
        scaled_offset = torch.zeros_like(self.base_offset*scale1)
        scaled_offset[:,self.mask1]  = self.base_offset[:,self.mask1] * scale1
        scaled_offset[:,self.mask2]  = self.base_offset[:,self.mask2] * scale2
        # scaled_offset = torch.einsum('badhw, bndhw->bnahw', basic_rot, scaled_offset.reshape(bs,-1,2,h,w)).reshape(bs,-1,h,w)
        
        offset =  (scaled_offset - self.base_offset)  + shift
        
        # offset[:, self.mask1] = shift[:,self.mask1] - self.base_offset[:,self.mask1] * (1 - scale1)
        # offset[:, self.mask2] = shift[:,self.mask2] -self.base_offset[:,self.mask2] * (1 - scale2)
        # offset = torch.einsum('badhw, bndhw->bnahw', basic_rot, offset.reshape(bs,-1,2,h,w)).reshape(bs,-1,h,w)
        
        x = ops.deform_conv2d(x, offset, _filter, _bias, stride=self.conv.stride, padding=self.conv.padding)#, mask=modulator)
        return x
    

class ExportTriEquiDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = None
        self.lifting = None
        self.act = nn.ReLU(inplace=True)
        self.out_layer = None
        self.fc_c = None
        self.blocks = None
        self.in_dim = None
        self.hidden_dim = None
        self.concat_dim = None
        self.n_blocks = None
        self.feature_sampler = None
        self.fc_p = None
        self.padding = 0
        self.sample_mode = 'bilinear'
        
    
    def query_feature(self, p, c_plane):
        if self.feature_sampler is not None:
            c = self.feature_sampler(p, c_plane)
        else:
            c = 0
            if 'grid' in c_plane:
                c = self.sample_grid_feature(p, c_plane['grid'])
                c = c.transpose(1,2)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
    
    def forward(self, p, c_plane):
        batch_size, sample_size = p.shape[0], p.shape[1]
        c = self.query_feature(p, c_plane)
        valid_mask = ~(torch.isnan(p).all(-1))
        

        c = c[valid_mask].reshape(-1, self.in_dim)
        c = self.act(self.lifting(c))

        if self.concat_dim is not None:
            net = p[valid_mask][...,3:].reshape(-1, self.concat_dim)
            net = self.fc_p(net)
        else:
            net = torch.zeros(c.shape[0], self.hidden_dim).to(p.device)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        net = self.out_layer(net)
        out = torch.full((batch_size, sample_size, net.shape[-1]), float('nan')).to(net.device)
        out[valid_mask] = net
        return out.squeeze(-1)

class ExportTimeEquiDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.N = None
        self.lifting = None
        self.act = nn.ReLU(inplace=True)
        self.out_layer = None
        self.fc_c = None
        self.blocks = None
        self.in_dim = None
        self.hidden_dim = None
        self.concat_dim = None
        self.n_blocks = None
        self.feature_sampler = None
        self.fc_p = None
        self.padding = 0
        self.sample_mode = 'bilinear'
        self.time_mlp = None
        self.t_dim = None
    
    def query_feature(self, p, c_plane, extra_data=None):
        c = self.feature_sampler(p, c_plane)
        return c
    
    def forward(self, data, c_plane, time):
        if isinstance(c_plane, dict):
            c = self.query_feature(data, c_plane)
        else:
            c = c_plane
        batch_size, sample_size = data.shape[0], data.shape[1]
        if len(time.shape) == 1:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size, 1,-1).repeat(1, sample_size, 1).reshape(batch_size*sample_size, self.t_dim)
        else:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size*sample_size,-1)
        

        c = c.reshape(-1,self.in_dim)
        c = self.act(self.lifting(c))
        net = data.reshape(batch_size*sample_size, -1)
        net = self.fc_p(net)
        
        c = torch.cat((c, t), dim=-1)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        net = self.out_layer(net)
        net = net.reshape(batch_size, sample_size, -1).squeeze(-1)
        return net

class ExportEquiDeformableAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_sampler = None
        self.num_samples = None
        self.to_out_indim = None
        self.in_dim = None
        self.to_offset = None
        self.to_weight = None
        self.to_v = None
        self.to_out = None
        
    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 3)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        
        feature = self.feature_sampler(query_pos, c)

        feature = feature.reshape(bs*ns, -1)

        aux_sample_point_offset = self.to_offset(feature).reshape(bs,ns,-1, 3) # (bs, ns, sp*sp*sp, 3)
        weight_offset = self.to_weight(feature).reshape(bs*ns, self.num_samples, 1) # (bs, ns, sp*sp*sp, 1)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 3)
        # return enn.GeometricTensor(aux_sample_point.reshape(-1, self.to_offset.out_type.size), self.to_offset.out_type)
        
        self.sample_points = aux_sample_point
                
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns*self.num_samples, self.in_dim) # (bs*ns*sp*sp*sp, feature_dim)
        
        
        # return aux_feature_offset
        v = self.to_v(aux_feature_offset).reshape(bs*ns, self.num_samples, -1) # (bs*ns, sp*sp*sp, embed_dim)
        
        out = (v * weight_offset).sum(-2)
        
        out = out.reshape(-1, self.to_out_indim)
        out = self.to_out(out)
        ##### attention part #####

        out = out + feature
        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out.reshape(bs, ns, -1)

class ExportEquiGraspSO3DeformableAttn2(nn.Module):
    def __init__(self,):
        super().__init__()
        self.num_heads = None
        self.feature_dim = None
        self.feature_sampler = None
        self.to_v = None
        self.to_out = None
        self.ncp = None
        self.to_weight = None      
        self.control_points = None
        self.zero_offset = None
        self.hidden_size = None
    
    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:] # should be rot_6d
        query_pos = query_pos[...,:3]
        assert query_ori.shape[-1] == 6
        
        # with torch.no_grad():
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        # rotation
        rot_SO3 = rotation_6d_to_matrix(query_ori).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        # control_points_ = control_points.clone()
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        # control_points = torch.einsum('bndp,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

        # print_grasp(voxel_grid[0].detach().cpu().numpy(), rot_SO3[0,0].detach().cpu().numpy(), (query_pos[0,0].detach().cpu().numpy()+0.5)*0.3)

                
        # feature fusion for offset
        if self.zero_offset:
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            anchor_sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
        
        

        sample_feature = sample_feature.reshape(-1, self.feature_dim)
        feature = feature.reshape(-1, self.feature_dim)
        weights = self.to_weight(feature).reshape(bs*ns, self.ncp, self.num_heads, 1) # (bs*ns, ncp, n_heads, 1)
        # return feature
        
        v = self.to_v(sample_feature).reshape(bs*ns,  -1, self.num_heads, self.hidden_size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        
        out = (v * weights).sum((1,2))
        out = self.to_out(out)#.tensor.reshape(bs, ns, -1)
        out += feature#.tensor.reshape(out.shape)

        return out.reshape(bs, ns, -1)
    
    
    
class ExportEquiGraspSO3DeformableAttn(nn.Module):
    def __init__(self,):
        super().__init__()
        self.num_heads = None
        self.feature_dim = None
        self.feature_sampler = None
        self.to_v = None
        self.to_q = None
        self.to_k = None
        self.to_out = None
        self.ncp = None
        self.to_weight = None      
        self.control_points = None
        self.zero_offset = None
        self.hidden_size = None
        self.pe_q = None
        self.pe_k = None
    
    
    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:] # should be rot_6d
        query_pos = query_pos[...,:3]
        assert query_ori.shape[-1] == 6
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        # rotation
        rot_SO3 = rotation_6d_to_matrix(query_ori).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

                
        # feature fusion for offset
        if self.zero_offset:
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            anchor_sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
        
        

        sample_feature = sample_feature.reshape(-1, self.feature_dim)
        feature = feature.reshape(-1, self.feature_dim)
        # return feature
        
        k = self.to_k(sample_feature).tensor.reshape(bs*ns,  -1, self.num_heads, self.hidden_size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        v = self.to_v(sample_feature).tensor.reshape(bs*ns,  -1, self.num_heads, self.hidden_size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        
        q = self.to_q(feature).tensor.reshape(bs*ns, 1, self.hidden_size).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q + self.pe_q.reshape(1,1,1,-1)
        k = k + self.pe_k.reshape(1, 1, self.ncp, -1)

        
        q = q / self.scale
        
        sim = torch.einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        # attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = torch.einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs*ns,-1) # (bs, ns, n_heads*embed_dim)
        
        out = self.to_out(out)#.tensor.reshape(bs, ns, -1)
        out += feature#.tensor.reshape(out.shape)
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out.reshape(bs, ns, -1)