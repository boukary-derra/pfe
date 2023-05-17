def get_li(self):
    """ -> LI plays a significant role in differentiating target motion
        from background motion.
        -> The new LIM that considers velocity and motion direction """
    # get f_on, f_off
    # f_on, f_off = self.get_hwr()
    f_on, f_off = self.get_hwr()

    # get motion_vectors
    motion_vectors = block_based_motion_estimation(self.frame, self.last_frame, self.block_size, self.search_range)

    # get u, v
    u, v = get_motion_components(motion_vectors)

    # get h
    h = create_convolution_kernel(self.p, self.q, self.a)

    # get u_c, v_c
    u_c, v_c = convolve_uv_with_h(u, v, h)

    # calculate w
    w = calculate_w(u_c, v_c)
    f_on_li = self.k1 * f_on + self.k2 * f_on * w
    f_off_li = self.k1 * f_off + self.k2 * f_off * w

    return f_on_li, f_off_li
