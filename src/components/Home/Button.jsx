import { Link } from "react-scroll";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight } from "@fortawesome/free-solid-svg-icons";

function Button(props) {
  return (
    <Link
      to="contact"
      smooth={true}
      offset={-50}
      className="r cursor-pointer flex items-center gap-3 justify-center mt-5 w-[172px] h-[36px] text-[13px] lg:w-[286px] lg:h-[60px] lg:text-[22px] bg-[#FFFFFF] text-[#000] rounded-[8px] hover:bg-[#0049FF] hover:text-white transition-all border-[1px] border-[#fff]"
    >
      {props.title}{" "}
      <FontAwesomeIcon
        icon={faArrowRight}
        className="w-[16px] lg:w-[24px] h-[16px] lg:h-[24px]"
      />
    </Link>
  );
}
export default Button;
