function Header(props) {
  return (
    <div>
      <p className="text-[16px] leading-tight text-[#777E90] b">
        {props.header}
      </p>
      <div className="w-full h-px bg-[#D9D9D9] "></div>
    </div>
  );
}
export default Header;
